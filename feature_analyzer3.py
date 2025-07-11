import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import time
from scipy import stats

class FeatureAnalyzer:
    def __init__(self, csv_path, name="Features"):
        self.csv_path = csv_path
        self.name = name
        self.df = None
        self.numeric_cols = []
        self.feature_cols = []
        self.load_data()
    
    def load_data(self):
        """Load CSV data with retry logic for containerized environment"""
        max_retries = 10
        for attempt in range(max_retries):
            try:
                if os.path.exists(self.csv_path):
                    self.df = pd.read_csv(self.csv_path)
                    self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
                    self.feature_cols = [col for col in self.numeric_cols if 'Id' not in col and 'nodeId' not in col]
                    print(f"✅ Loaded {self.name} data: {self.df.shape}")
                    return
                else:
                    print(f"⏳ Waiting for {self.csv_path} (attempt {attempt + 1}/{max_retries})")
                    time.sleep(5)
            except Exception as e:
                print(f"❌ Error loading {self.csv_path}: {e}")
                time.sleep(5)
        
        # Create dummy data if file not found
        print(f"⚠️ Could not load {self.csv_path}, creating dummy data")
        self.df = pd.DataFrame({'nodeId': [1, 2, 3], 'dummy_feature': [0.1, 0.2, 0.3]})
        self.numeric_cols = ['dummy_feature']
        self.feature_cols = ['dummy_feature']

    def refresh_data(self):
        """Refresh data from CSV - useful for containerized apps"""
        self.load_data()
        return f"🔄 Data refreshed for {self.name}. Shape: {self.df.shape if self.df is not None else 'No data'}"
    
    def get_feature_stats(self):
        """Calculate comprehensive statistics for all features"""
        if self.df is None or len(self.feature_cols) == 0:
            return pd.DataFrame({"Error": ["No data available"]})
            
        stats_df = self.df[self.feature_cols].describe().T
        
        # Add additional metrics
        stats_df['variance'] = self.df[self.feature_cols].var()
        stats_df['skewness'] = self.df[self.feature_cols].skew()
        stats_df['kurtosis'] = self.df[self.feature_cols].kurtosis()
        stats_df['unique_values'] = self.df[self.feature_cols].nunique()
        stats_df['zero_count'] = (self.df[self.feature_cols] == 0).sum()
        stats_df['zero_percentage'] = (self.df[self.feature_cols] == 0).sum() / len(self.df) * 100
        
        # Calculate coefficient of variation (spread relative to mean)
        stats_df['cv'] = stats_df['std'] / stats_df['mean']
        
        # Reset index to make feature names a column
        stats_df = stats_df.reset_index()
        stats_df = stats_df.rename(columns={'index': 'Feature'})
        
        return stats_df.round(4)
    
    def create_boxplot(self, selected_features):
        """Create boxplots for selected features"""
        if not selected_features or self.df is None:
            return None
            
        n_features = len(selected_features)
        fig, axes = plt.subplots(1, min(n_features, 4), figsize=(15, 6))
        
        if n_features == 1:
            axes = [axes]
        
        for i, feature in enumerate(selected_features[:4]):  # Limit to 4 plots
            if i < len(axes) and feature in self.df.columns:
                self.df.boxplot(column=feature, ax=axes[i])
                axes[i].set_title(f'{feature}')
                axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig
    
    def create_distribution_plot(self, selected_features):
        """Create distribution plots for selected features"""
        if not selected_features or self.df is None:
            return None
            
        n_features = len(selected_features)
        fig, axes = plt.subplots(2, min(n_features, 2), figsize=(12, 8))
        
        if n_features == 1:
            axes = axes.reshape(-1)
        
        for i, feature in enumerate(selected_features[:4]):
            if feature not in self.df.columns:
                continue
                
            row = i // 2
            col = i % 2
            
            if n_features == 1:
                ax = axes[0] if i < 2 else axes[1]
            else:
                ax = axes[row, col] if n_features > 2 else axes[i]
            
            # Histogram with KDE
            self.df[feature].hist(bins=30, alpha=0.7, ax=ax, density=True)
            self.df[feature].plot.kde(ax=ax, color='red')
            ax.set_title(f'{feature} Distribution')
            ax.set_xlabel(feature)
            ax.set_ylabel('Density')
        
        plt.tight_layout()
        return fig
    
    def create_correlation_heatmap(self, selected_features):
        """Create correlation heatmap for selected features"""
        if len(selected_features) < 2 or self.df is None:
            return None
            
        correlation_matrix = self.df[selected_features].corr()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, ax=ax, fmt='.2f')
        ax.set_title('Feature Correlation Heatmap')
        plt.tight_layout()
        return fig
    
    def analyze_feature_quality(self, selected_features):
        """Analyze feature quality and provide recommendations"""
        if not selected_features or self.df is None:
            return "Please select features to analyze or check if data is loaded."
        
        analysis = []
        stats_df = self.get_feature_stats()
        
        for feature in selected_features:
            # Find the feature in the stats dataframe
            feature_row = stats_df[stats_df['Feature'] == feature]
            if feature_row.empty:
                continue
                
            row = feature_row.iloc[0]
            
            # Feature quality analysis
            quality_score = 0
            issues = []
            recommendations = []
            
            # Check variance (low variance = low information)
            if row['std'] < 0.01:
                issues.append("Very low variance")
            elif row['cv'] < 0.1:
                issues.append("Low coefficient of variation")
                quality_score += 1
            else:
                quality_score += 2
            
            # Check for too many zeros
            if row['zero_percentage'] > 80:
                issues.append("Too many zeros (>80%)")
            elif row['zero_percentage'] > 50:
                issues.append("Many zeros (>50%)")
                quality_score += 1
            else:
                quality_score += 2
            
            # Check unique values
            if row['unique_values'] < 3:
                issues.append("Too few unique values")
            else:
                quality_score += 1
            
            # Check skewness
            if abs(row['skewness']) > 2:
                recommendations.append("Consider log transformation (high skewness)")
            
            # Overall assessment
            if quality_score >= 4:
                assessment = "✅ Good feature"
            elif quality_score >= 2:
                assessment = "⚠️ Moderate feature" 
            else:
                assessment = "❌ Poor feature"
            
            analysis.append(f"""
                **{feature}**: {assessment}
                - Mean: {row['mean']:.4f}, Std: {row['std']:.4f}
                - CV: {row['cv']:.4f}, Unique values: {int(row['unique_values'])}
                - Zero percentage: {row['zero_percentage']:.1f}%
                - Issues: {', '.join(issues) if issues else 'None'}
                - Recommendations: {', '.join(recommendations) if recommendations else 'None'}
                            """)
        
        return "\n".join(analysis)

    def get_structural_features_explanation(self):
        """Get explanation for structural features"""
        return """
                ## 📖 Understanding Structural Features

                ### Common Network Metrics:
                - **PageRank**: Measures node importance based on incoming connections and their quality (0-1 scale, higher = more important)
                - **Betweenness Centrality**: Measures how often a node lies on shortest paths between other nodes (0-1 scale, higher = more central)
                - **Closeness Centrality**: Measures how close a node is to all other nodes in the network (0-1 scale, higher = more central)
                - **Degree Centrality**: Measures the fraction of nodes a node is connected to (0-1 scale, higher = more connected)
                - **Eigenvector Centrality**: Measures influence based on connections to other influential nodes (0-1 scale, higher = more influential)

                ### Statistical Shape Measures:
                - **Skewness**: Measures asymmetry of the distribution
                - **Scale**: Typically ranges from -3 to +3
                - **Interpretation**: 
                    - Close to 0: Symmetric distribution
                    - Positive (>0): Right-skewed (tail extends right)
                    - Negative (<0): Left-skewed (tail extends left)
                    - |Skewness| > 2: Highly skewed, may need transformation

                - **Kurtosis**: Measures "tailedness" of the distribution
                - **Scale**: Typically ranges from -2 to +10 (excess kurtosis)
                - **Interpretation**:
                    - Close to 0: Normal distribution-like tails
                    - Positive: Heavy tails (more extreme values)
                    - Negative: Light tails (fewer extreme values)
                    - |Kurtosis| > 3: Very different from normal distribution
                        """

    def get_quality_assessment_explanation(self):
        """Get explanation for quality assessment methodology"""
        return """
                    ## 🎯 Quality Assessment Methodology

                    ### Coefficient of Variation (CV):
                    **Formula**: CV = Standard Deviation / Mean
                    - **Purpose**: Measures relative variability (spread relative to the mean)
                    - **Interpretation**:
                    - CV < 0.1: Low variability (may lack discriminative power)
                    - CV 0.1-0.3: Moderate variability (good for analysis)
                    - CV > 0.3: High variability (good discriminative power)

                    ### Quality Scoring System:
                    **Each feature gets 0-5 points based on:**
                    1. **Variance Check** (0-2 points):
                    - Very low variance (<0.01): 0 points - Feature barely varies
                    - Low CV (<0.1): 1 point - Limited discriminative power
                    - Good CV (≥0.1): 2 points - Good discriminative power

                    2. **Zero Analysis** (0-2 points):
                    - >80% zeros: 0 points - Too sparse, may be unreliable
                    - 50-80% zeros: 1 point - Moderately sparse
                    - <50% zeros: 2 points - Good data density

                    3. **Unique Values** (0-1 points):
                    - <3 unique values: 0 points - Too few categories
                    - ≥3 unique values: 1 point - Sufficient variation

                    ### Issue Detection:
                    - **Low Variance**: Feature doesn't vary much across samples
                    - **Too Many Zeros**: Feature may be sparse or have data quality issues
                    - **High Skewness**: Distribution is highly asymmetric (may need transformation)

                    ### Recommendations:
                    - **Log Transformation**: Suggested for highly skewed data (|skewness| > 2)
                    - **Normalization**: May be needed for features with very different scales
                    - **Feature Engineering**: Consider combining or transforming poorly performing features

                    ### Overall Assessment:
                    - **✅ Good Feature** (4-5 points): Ready for analysis
                    - **⚠️ Moderate Feature** (2-3 points): Usable but may need preprocessing
                    - **❌ Poor Feature** (0-1 points): Consider removal or major transformation
                            """


class GradioApp:
    def __init__(self, csv_dir='/csv_output'):
        self.csv_dir = csv_dir
        self.structural_analyzer = FeatureAnalyzer(f'{csv_dir}/structural_features.csv', name="Structural")
        self.community_analyzer = FeatureAnalyzer(f'{csv_dir}/community_features.csv', name="Community")
        self.pathfinding_analyzer = FeatureAnalyzer(f'{csv_dir}/pathfinding_features.csv', name="Pathfinding")
    
    def create_gradio_app(self):
        """Create standalone Gradio app for analyzing CSV files"""
        with gr.Blocks(title="Pest Analysis Feature Dashboard") as demo:
            gr.Markdown("# 📊 Pest Data Feature Analysis Dashboard")
            gr.Markdown("*Monitoring for CSV files from main analysis...*")
            
            with gr.Tab("🏗️ Structural Features"):
                # Add refresh button
                with gr.Row():
                    refresh_struct_btn = gr.Button("🔄 Refresh Data")
                    refresh_struct_output = gr.Textbox(label="Status", interactive=False)
                
                refresh_struct_btn.click(
                    self.structural_analyzer.refresh_data, 
                    outputs=refresh_struct_output
                )
                
                # Add structural features explanation
                with gr.Accordion("📖 Understanding Structural Features", open=False):
                    gr.Markdown(self.structural_analyzer.get_structural_features_explanation())
                
                self._create_analysis_tab(self.structural_analyzer)
            
            with gr.Tab("🏘️ Community Features"):
                # Add refresh button
                with gr.Row():
                    refresh_comm_btn = gr.Button("🔄 Refresh Data")
                    refresh_comm_output = gr.Textbox(label="Status", interactive=False)
                
                refresh_comm_btn.click(
                    self.community_analyzer.refresh_data, 
                    outputs=refresh_comm_output
                )
                
                self._create_analysis_tab(self.community_analyzer)
            
            with gr.Tab("🔍 Pathfinding Features"):
                # Add refresh button
                with gr.Row():
                    refresh_path_btn = gr.Button("🔄 Refresh Data")
                    refresh_path_output = gr.Textbox(label="Status", interactive=False)
                
                refresh_path_btn.click(
                    self.pathfinding_analyzer.refresh_data, 
                    outputs=refresh_path_output
                )
                
                self._create_analysis_tab(self.pathfinding_analyzer)
            
            with gr.Tab("🔀 Combined Analysis"):
                self._create_combined_analysis_tab()
        
        return demo
    
    def _create_analysis_tab(self, analyzer):
        """Create analysis tab for a specific feature type"""
        with gr.Tab("📈 Statistics"):
            stats_btn = gr.Button(f"Get {analyzer.name} Statistics")
            stats_output = gr.Dataframe()
            stats_btn.click(analyzer.get_feature_stats, outputs=stats_output)
        
        with gr.Tab("📦 Distributions"):
            feature_selector = gr.CheckboxGroup(
                choices=analyzer.feature_cols,
                label="Select Features (max 4)",
                value=analyzer.feature_cols[:4] if analyzer.feature_cols else []
            )
            
            with gr.Row():
                boxplot_btn = gr.Button("Generate Boxplots")
                dist_btn = gr.Button("Generate Distribution Plots")
            
            with gr.Row():
                boxplot_output = gr.Plot()
                dist_output = gr.Plot()
            
            boxplot_btn.click(analyzer.create_boxplot, inputs=feature_selector, outputs=boxplot_output)
            dist_btn.click(analyzer.create_distribution_plot, inputs=feature_selector, outputs=dist_output)
        
        with gr.Tab("🎯 Quality Assessment"):
            # Add quality assessment explanation
            with gr.Row():
                with gr.Column(scale=1):
                    quality_selector = gr.CheckboxGroup(
                        choices=analyzer.feature_cols,
                        label="Select Features to Assess",
                        value=analyzer.feature_cols[:10] if analyzer.feature_cols else []
                    )
                    quality_btn = gr.Button("Analyze Feature Quality")
                    quality_output = gr.Markdown()
                
                with gr.Column(scale=1):
                    with gr.Accordion("📚 Quality Assessment Guide", open=True):
                        gr.Markdown(analyzer.get_quality_assessment_explanation())
            
            quality_btn.click(analyzer.analyze_feature_quality, inputs=quality_selector, outputs=quality_output)
    
    def _create_combined_analysis_tab(self):
        """Create tab for comparing all three feature types"""
        gr.Markdown("## 🔀 Compare All Feature Types")
        
        with gr.Row():
            with gr.Column():
                gr.Markdown("### Structural Features")
                struct_selector = gr.CheckboxGroup(
                    choices=self.structural_analyzer.feature_cols[:5] if self.structural_analyzer.feature_cols else [],
                    label="Select Structural Features",
                    value=self.structural_analyzer.feature_cols[:3] if self.structural_analyzer.feature_cols else []
                )
            
            with gr.Column():
                gr.Markdown("### Community Features") 
                comm_selector = gr.CheckboxGroup(
                    choices=self.community_analyzer.feature_cols[:5] if self.community_analyzer.feature_cols else [],
                    label="Select Community Features",
                    value=self.community_analyzer.feature_cols[:3] if self.community_analyzer.feature_cols else []
                )
            
            with gr.Column():
                gr.Markdown("### Pathfinding Features")
                path_selector = gr.CheckboxGroup(
                    choices=self.pathfinding_analyzer.feature_cols[:5] if self.pathfinding_analyzer.feature_cols else [],
                    label="Select Pathfinding Features",
                    value=self.pathfinding_analyzer.feature_cols[:3] if self.pathfinding_analyzer.feature_cols else []
                )
        
        compare_btn = gr.Button("Compare All Feature Types")
        compare_output = gr.Markdown()
        
        compare_btn.click(self._compare_all_features, inputs=[struct_selector, comm_selector, path_selector], outputs=compare_output)
    
    def _compare_all_features(self, struct_features, comm_features, path_features):
        """Compare features across all three types"""
        result = f"## 📊 Comprehensive Feature Comparison\n\n"
        
        # Analyze structural features
        if struct_features:
            struct_analysis = self.structural_analyzer.analyze_feature_quality(struct_features)
            result += f"### 🏗️ Structural Features Analysis\n{struct_analysis}\n\n"
        
        # Analyze community features  
        if comm_features:
            comm_analysis = self.community_analyzer.analyze_feature_quality(comm_features)
            result += f"### 🏘️ Community Features Analysis\n{comm_analysis}\n\n"
        
        # Analyze pathfinding features
        if path_features:
            path_analysis = self.pathfinding_analyzer.analyze_feature_quality(path_features)
            result += f"### 🔍 Pathfinding Features Analysis\n{path_analysis}\n\n"
        
        return result


if __name__ == "__main__":
    print("🚀 Starting Feature Analyzer App...")
    print("📂 Looking for CSV files in /csv_output/")
    
    # Ensure output directory exists
    os.makedirs('/csv_output', exist_ok=True)
    
    # Create and launch the app
    app = GradioApp()
    demo = app.create_gradio_app()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
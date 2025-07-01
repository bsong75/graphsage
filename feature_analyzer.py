import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

class FeatureAnalyzer:
    def __init__(self, csv_path, name="Features"):
        self.df = pd.read_csv(csv_path)
        self.name = name
        self.numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        # Remove ID columns from analysis
        self.feature_cols = [col for col in self.numeric_cols if 'Id' not in col and 'nodeId' not in col]
    
    def get_feature_stats(self):
        """Calculate comprehensive statistics for all features"""
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
        
        return stats_df.round(4)
    
    def create_boxplot(self, selected_features):
        """Create boxplots for selected features"""
        if not selected_features:
            return None
            
        n_features = len(selected_features)
        fig, axes = plt.subplots(1, min(n_features, 4), figsize=(15, 6))
        
        if n_features == 1:
            axes = [axes]
        
        for i, feature in enumerate(selected_features[:4]):  # Limit to 4 plots
            if i < len(axes):
                self.df.boxplot(column=feature, ax=axes[i])
                axes[i].set_title(f'{feature}')
                axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig
    
    def create_distribution_plot(self, selected_features):
        """Create distribution plots for selected features"""
        if not selected_features:
            return None
            
        n_features = len(selected_features)
        fig, axes = plt.subplots(2, min(n_features, 2), figsize=(12, 8))
        
        if n_features == 1:
            axes = axes.reshape(-1)
        
        for i, feature in enumerate(selected_features[:4]):
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
        if len(selected_features) < 2:
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
        if not selected_features:
            return "Please select features to analyze."
        
        analysis = []
        stats_df = self.get_feature_stats()
        
        for feature in selected_features:
            if feature not in stats_df.index:
                continue
                
            row = stats_df.loc[feature]
            
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


def create_gradio_dashboard(csv_files):
    """Create Gradio dashboard for multiple CSV files"""
    analyzers = {}
    
    for csv_file in csv_files:
        if 'structural' in csv_file.lower():
            analyzers['structural'] = FeatureAnalyzer(csv_file, name="Structural")
        elif 'community' in csv_file.lower():
            analyzers['community'] = FeatureAnalyzer(csv_file, name="Community")
    
    with gr.Blocks(title="Pest Analysis Feature Dashboard") as demo:
        gr.Markdown("# 📊 Pest Data Feature Analysis Dashboard")
        
        if 'structural' in analyzers:
            with gr.Tab("🏗️ Structural Features"):
                _create_analysis_tab(analyzers['structural'])
        
        if 'community' in analyzers:
            with gr.Tab("🏘️ Community Features"):
                _create_analysis_tab(analyzers['community'])
        
        if len(analyzers) > 1:
            with gr.Tab("🔀 Combined Analysis"):
                _create_combined_analysis_tab(analyzers.get('structural'), analyzers.get('community'))
    
    return demo


def _create_analysis_tab(analyzer):
    """Create analysis tab for a specific feature type"""
    with gr.Tab("📈 Statistics"):
        stats_btn = gr.Button(f"Get {analyzer.name} Statistics")
        stats_output = gr.Dataframe()
        stats_btn.click(analyzer.get_feature_stats, outputs=stats_output)
    
    with gr.Tab("📦 Distributions"):
        feature_selector = gr.CheckboxGroup(
            choices=analyzer.feature_cols,
            label="Select Features (max 4)",
            value=analyzer.feature_cols[:4]
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
        quality_selector = gr.CheckboxGroup(
            choices=analyzer.feature_cols,
            label="Select Features to Assess",
            value=analyzer.feature_cols[:10]
        )
        quality_btn = gr.Button("Analyze Feature Quality")
        quality_output = gr.Markdown()
        quality_btn.click(analyzer.analyze_feature_quality, inputs=quality_selector, outputs=quality_output)


def _create_combined_analysis_tab(analyzer1, analyzer2):
    """Create tab for comparing structural vs community features"""
    if not analyzer1 or not analyzer2:
        gr.Markdown("Combined analysis requires both structural and community features.")
        return
    
    gr.Markdown("## 🔀 Compare Structural vs Community Features")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Structural Features")
            struct_selector = gr.CheckboxGroup(
                choices=analyzer1.feature_cols[:5],
                label="Select Structural Features",
                value=analyzer1.feature_cols[:3]
            )
        
        with gr.Column():
            gr.Markdown("### Community Features") 
            comm_selector = gr.CheckboxGroup(
                choices=analyzer2.feature_cols[:5],
                label="Select Community Features",
                value=analyzer2.feature_cols[:3]
            )
    
    compare_btn = gr.Button("Compare Feature Quality")
    compare_output = gr.Markdown()
    
    def compare_features(struct_features, comm_features):
        result = f"## 📊 Feature Comparison\n\n"
        
        # Analyze structural features
        if struct_features:
            struct_analysis = analyzer1.analyze_feature_quality(struct_features)
            result += f"### 🏗️ Structural Features Analysis\n{struct_analysis}\n\n"
        
        # Analyze community features  
        if comm_features:
            comm_analysis = analyzer2.analyze_feature_quality(comm_features)
            result += f"### 🏘️ Community Features Analysis\n{comm_analysis}\n\n"
        
        return result
    
    compare_btn.click(compare_features, inputs=[struct_selector, comm_selector], outputs=compare_output)
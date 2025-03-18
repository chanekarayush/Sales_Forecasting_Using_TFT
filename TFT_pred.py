import pandas as pd
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import SMAPE, MAE, RMSE
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import streamlit as st
import io
import plotly.express as px
import plotly.graph_objects as go


class DemandPredictionSystem:
    def __init__(self):
        self.df = None
        self.training_cutoff = None
        self.max_encoder_length = 4  # Default to 1 year (4 quarters)
        self.max_prediction_length = 1  # Default to 1 quarter ahead prediction
        self.model = None
        self.training = None
        self.validation = None
        self.trainer = None

    def load_data(self, uploaded_file):
        """Load and preprocess the uploaded CSV file"""
        if uploaded_file is not None:
            self.df = pd.read_csv(uploaded_file)
            return self.preprocess_data()
        return False

    def preprocess_data(self):
        """Preprocess the data for the TFT model"""
        try:
            # Ensure required columns exist
            required_cols = ['distributor_id', 'sku', 'category', 'sales',
                             'quarter', 'year', 'time_idx']

            for col in required_cols:
                if col not in self.df.columns:
                    st.error(
                        f"Required column '{col}' not found in the uploaded data")
                    return False

            # Create time index if not present
            if 'time_idx' not in self.df.columns:
                self.df = self.df.sort_values(
                    by=['distributor_id', 'sku', 'year', 'quarter'])
                self.df['time_idx'] = self.df.groupby(
                    ['distributor_id', 'sku']).cumcount() + 1

            # Create categorical variables if missing
            if 'movement_category' not in self.df.columns:
                self.df['movement_category'] = 'Unknown'

            if 'industry' not in self.df.columns:
                self.df['industry'] = 'Unknown'

            # Create quarter-year field for better visualization
            self.df['quarter_year'] = self.df['year'].astype(
                str) + "-Q" + self.df['quarter'].astype(str)

            # Set training cutoff at 80% of the data
            self.training_cutoff = int(self.df["time_idx"].max() * 0.8)

            return True

        except Exception as e:
            st.error(f"Error in preprocessing data: {str(e)}")
            return False

    def create_datasets(self):
        """Create training and validation datasets for the TFT model"""
        # Define the training dataset
        training = TimeSeriesDataSet(
            data=self.df[self.df["time_idx"] <= self.training_cutoff],
            time_idx="time_idx",
            target="sales",
            group_ids=["distributor_id", "sku"],
            max_encoder_length=self.max_encoder_length,
            max_prediction_length=self.max_prediction_length,
            static_categoricals=["industry", "category", "movement_category"],
            time_varying_known_categoricals=["quarter"],
            time_varying_known_reals=["year"],
            time_varying_unknown_reals=["sales", "avg_quarterly_sales",
                                        "total_quarter_sales", "prev_quarter_sales"],
            target_normalizer=GroupNormalizer(
                groups=["distributor_id", "sku"], transformation="softplus"
            ),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )

        # Create validation set using the same parameters as the training set
        validation = TimeSeriesDataSet.from_dataset(
            training, self.df, min_prediction_idx=self.training_cutoff + 1
        )

        self.training = training
        self.validation = validation

        return training, validation

    def train_model(self):
        """Train the TFT model"""
        try:
            train_dataloader = self.training.to_dataloader(
                batch_size=32, num_workers=0, shuffle=True)
            val_dataloader = self.validation.to_dataloader(
                batch_size=32, num_workers=0, shuffle=False)

            # Configure the model
            tft = TemporalFusionTransformer.from_dataset(
                self.training,
                learning_rate=0.03,
                hidden_size=32,
                attention_head_size=1,
                dropout=0.1,
                hidden_continuous_size=16,
                loss=SMAPE(),
                log_interval=10,
                reduce_on_plateau_patience=3,
            )

            # Configure trainer with early stopping
            early_stop_callback = EarlyStopping(
                monitor="val_loss", min_delta=1e-4, patience=5, verbose=False, mode="min"
            )
            lr_logger = LearningRateMonitor()

            trainer = pl.Trainer(
                max_epochs=15,
                accelerator="auto",
                gradient_clip_val=0.1,
                limit_train_batches=50,
                callbacks=[early_stop_callback, lr_logger],
            )

            # Train the model
            trainer.fit(
                tft,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader,
            )

            self.model = tft
            self.trainer = trainer

            return tft, trainer

        except Exception as e:
            st.error(f"Error training model: {str(e)}")
            return None, None

    def make_predictions(self, future_periods=1):
        """Make predictions for future periods"""
        if self.model is None:
            st.error("Model needs to be trained first")
            return None

        # Create prediction dataset
        encoder_data = self.df[
            self.df.time_idx > self.df.time_idx.max() - self.max_encoder_length
        ]

        # Get the last time_idx in the dataset
        last_time_idx = self.df.time_idx.max()

        # Create a dataframe for future periods
        future_data = []

        # Get unique distributor-sku combinations
        unique_combinations = self.df[[
            'distributor_id', 'sku', 'category', 'industry', 'movement_category']].drop_duplicates()

        # For each unique combination
        for _, row in unique_combinations.iterrows():
            dist_id = row['distributor_id']
            sku = row['sku']
            category = row['category']
            industry = row['industry']
            movement = row['movement_category']

            # Get the last data point for this combination
            last_data = self.df[(self.df['distributor_id'] == dist_id) &
                                (self.df['sku'] == sku)].sort_values('time_idx').iloc[-1]

            # Create future periods
            for i in range(1, future_periods + 1):
                new_time_idx = last_time_idx + i
                new_year = last_data['year']
                new_quarter = last_data['quarter'] + i

                # Handle quarter rollover
                while new_quarter > 4:
                    new_quarter -= 4
                    new_year += 1

                # Placeholder values for sales metrics (will be predicted)
                new_row = {
                    'distributor_id': dist_id,
                    'sku': sku,
                    'category': category,
                    'industry': industry,
                    'movement_category': movement,
                    'quarter': new_quarter,
                    'year': new_year,
                    'time_idx': new_time_idx,
                    'quarter_year': f"{new_year}-Q{new_quarter}",
                    'sales': 0,  # Placeholder
                    'avg_quarterly_sales': last_data['avg_quarterly_sales'],
                    'total_quarter_sales': 0,  # Placeholder
                    'prev_quarter_sales': 0  # Placeholder
                }

                # Add festival indicators based on the quarter and year
                new_row['is_diwali'] = 1 if new_quarter == 4 else 0
                new_row['is_ganesh_chaturthi'] = 1 if new_quarter == 3 else 0
                new_row['is_gudi_padwa'] = 1 if new_quarter in [1, 2] else 0
                # Simplifying the logic for Eid
                new_row['is_eid'] = 1 if new_quarter in [1, 2] else 0
                new_row['is_akshay_tritiya'] = 1 if new_quarter == 2 else 0
                new_row['is_dussehra_navratri'] = 1 if new_quarter in [
                    3, 4] else 0
                new_row['is_onam'] = 1 if new_quarter == 3 else 0
                new_row['is_christmas'] = 1 if new_quarter == 4 else 0

                future_data.append(new_row)

        prediction_df = pd.DataFrame(future_data)

        # Create combined dataset with encoder data and future data
        combined_df = pd.concat([encoder_data, prediction_df])

        # Get predictions for the next quarter
        predictions, x = self.model.predict(combined_df, return_x=True)

        # Add predictions to the prediction dataframe
        prediction_df['predicted_sales'] = predictions.numpy().flatten()

        return prediction_df

    def generate_report(self, prediction_df):
        """Generate a comprehensive report based on predictions"""
        # Round predictions for better readability
        prediction_df['predicted_sales'] = prediction_df['predicted_sales'].round(
            2)

        # Calculate growth vs previous actuals
        latest_actuals = self.df.groupby(['distributor_id', 'sku']).apply(
            lambda x: x.sort_values('time_idx').iloc[-1]['sales']
        ).reset_index()
        latest_actuals.columns = ['distributor_id', 'sku', 'last_actual_sales']

        # Merge with predictions
        report_df = pd.merge(
            prediction_df,
            latest_actuals,
            on=['distributor_id', 'sku'],
            how='left'
        )

        # Calculate growth percentage
        report_df['growth_pct'] = ((report_df['predicted_sales'] - report_df['last_actual_sales']) /
                                   report_df['last_actual_sales'] * 100).round(2)

        # Add recommendations based on growth and movement category
        def get_recommendation(row):
            if row['growth_pct'] > 10:
                if row['movement_category'] in ['Fast Moving', 'Medium']:
                    return "Increase order quantity by 15-20%"
                else:
                    return "Consider slight increase in orders (5-10%)"
            elif row['growth_pct'] > 0:
                return "Maintain current order levels"
            else:
                if row['movement_category'] == 'Slow Moving':
                    return "Consider reducing order quantity by 10-15%"
                else:
                    return "Slight reduction in orders may be needed (5-10%)"

        report_df['recommendation'] = report_df.apply(
            get_recommendation, axis=1)

        # Add inventory risk assessment
        def assess_risk(row):
            if row['movement_category'] == 'Fast Moving' and row['growth_pct'] > 5:
                return "Low - High turnover expected"
            elif row['movement_category'] == 'Slow Moving' and row['growth_pct'] < 0:
                return "High - May lead to excess inventory"
            elif row['movement_category'] == 'Medium':
                return "Medium - Monitor closely"
            else:
                return "Medium - Standard risk"

        report_df['inventory_risk'] = report_df.apply(assess_risk, axis=1)

        return report_df


def run_streamlit_app():
    """Main function to run the Streamlit app"""
    st.title("Demand Prediction and Inventory Recommendation System")

    # Initialize the system
    system = DemandPredictionSystem()

    # File upload section
    st.header("1. Upload Sales Data")
    uploaded_file = st.file_uploader(
        "Upload CSV file with sales data", type=["csv"])

    if uploaded_file is not None:
        # Load and preprocess data
        if system.load_data(uploaded_file):
            st.success("Data loaded and preprocessed successfully!")

            # Display data summary
            st.subheader("Data Summary")
            st.write(f"Total records: {len(system.df)}")
            st.write(f"Distributors: {system.df['distributor_id'].nunique()}")
            st.write(f"Products: {system.df['sku'].nunique()}")
            st.write(f"Categories: {system.df['category'].nunique()}")
            st.write(
                f"Time range: {system.df['year'].min()}-Q{system.df['quarter'].min()} to {system.df['year'].max()}-Q{system.df['quarter'].max()}")

            # Show sample data
            st.subheader("Sample Data")
            st.dataframe(system.df.head())

            # Configure model parameters
            st.header("2. Configure Model Parameters")
            col1, col2 = st.columns(2)

            with col1:
                system.max_encoder_length = st.slider(
                    "Historical quarters to consider",
                    min_value=1,
                    max_value=8,
                    value=4,
                    help="Number of past quarters to use for predicting future sales"
                )

            with col2:
                prediction_periods = st.slider(
                    "Future quarters to predict",
                    min_value=1,
                    max_value=4,
                    value=1,
                    help="Number of future quarters to predict"
                )

            # Product categories selection for analysis
            st.subheader("Select Categories for Focus")
            selected_categories = st.multiselect(
                "Select product categories to analyze in detail",
                options=sorted(system.df['category'].unique()),
                default=sorted(system.df['category'].unique())[:3] if len(
                    system.df['category'].unique()) > 3 else sorted(system.df['category'].unique())
            )

            # Movement category mapping
            st.subheader("Define Movement Categories (Optional)")
            st.write("You can map existing movement categories or define new ones")

            # Show current movement categories
            unique_movements = system.df['movement_category'].unique()

            # Create movement category mapping interface
            st.write("Current movement categories in data:",
                     ", ".join(unique_movements))

            # Allow user to reclassify based on sales volume
            st.write("Reclassify products based on sales volume:")

            col1, col2, col3 = st.columns(3)

            with col1:
                fast_threshold = st.number_input(
                    "Fast Moving threshold (percentile)",
                    min_value=50.0,
                    max_value=99.0,
                    value=80.0,
                    step=5.0,
                    help="Products with sales above this percentile will be classified as Fast Moving"
                )

            with col2:
                slow_threshold = st.number_input(
                    "Slow Moving threshold (percentile)",
                    min_value=1.0,
                    max_value=50.0,
                    value=20.0,
                    step=5.0,
                    help="Products with sales below this percentile will be classified as Slow Moving"
                )

            with col3:
                apply_reclassification = st.checkbox(
                    "Apply reclassification",
                    value=False,
                    help="Check to reclassify movement categories based on thresholds"
                )

            if apply_reclassification:
                # Calculate average sales per SKU
                avg_sales = system.df.groupby(
                    'sku')['sales'].mean().reset_index()

                # Calculate percentiles
                fast_cutoff = np.percentile(avg_sales['sales'], fast_threshold)
                slow_cutoff = np.percentile(avg_sales['sales'], slow_threshold)

                # Create mapping dictionary
                movement_map = {}
                for _, row in avg_sales.iterrows():
                    if row['sales'] >= fast_cutoff:
                        movement_map[row['sku']] = 'Fast Moving'
                    elif row['sales'] <= slow_cutoff:
                        movement_map[row['sku']] = 'Slow Moving'
                    else:
                        movement_map[row['sku']] = 'Medium'

                # Apply mapping
                original_movement = system.df['movement_category'].copy()
                system.df['movement_category'] = system.df['sku'].map(
                    movement_map)

                st.write(
                    f"Reclassification completed: {(system.df['movement_category'] != original_movement).sum()} products reclassified")
                st.write(
                    f"New distribution: {system.df['movement_category'].value_counts().to_dict()}")

            # Train and predict button
            st.header("3. Train Model and Generate Predictions")

            if st.button("Train Model and Generate Predictions"):
                with st.spinner("Creating datasets..."):
                    system.create_datasets()
                    st.success("Datasets created!")

                with st.spinner("Training model... This may take a few minutes."):
                    model, trainer = system.train_model()
                    if model is not None:
                        st.success("Model training complete!")
                    else:
                        st.error(
                            "Error in model training. Please check your data and try again.")
                        st.stop()

                with st.spinner("Generating predictions..."):
                    predictions = system.make_predictions(
                        future_periods=prediction_periods)
                    if predictions is not None:
                        st.success("Predictions generated!")
                    else:
                        st.error("Error generating predictions.")
                        st.stop()

                # Generate report
                with st.spinner("Creating recommendation report..."):
                    report = system.generate_report(predictions)
                    st.success("Report generated!")

                # Display report
                st.header("4. Inventory Recommendations")

                # Filter by selected categories if specified
                if selected_categories:
                    filtered_report = report[report['category'].isin(
                        selected_categories)]
                else:
                    filtered_report = report

                # Summary statistics
                st.subheader("Prediction Summary")

                # Category breakdown
                cat_summary = filtered_report.groupby(
                    'category')[['predicted_sales', 'last_actual_sales']].sum().reset_index()
                cat_summary['growth_pct'] = ((cat_summary['predicted_sales'] - cat_summary['last_actual_sales']) /
                                             cat_summary['last_actual_sales'] * 100).round(2)

                # Plot category growth
                fig = px.bar(
                    cat_summary,
                    x='category',
                    y='growth_pct',
                    title='Predicted Sales Growth by Category',
                    labels={'growth_pct': 'Growth %',
                            'category': 'Product Category'},
                    color='growth_pct',
                    color_continuous_scale=['red', 'yellow', 'green'],
                    range_color=[-10, 20]
                )
                st.plotly_chart(fig)

                # Movement category insights
                move_summary = filtered_report.groupby('movement_category')[
                    ['predicted_sales', 'last_actual_sales']].sum().reset_index()
                move_summary['growth_pct'] = ((move_summary['predicted_sales'] - move_summary['last_actual_sales']) /
                                              move_summary['last_actual_sales'] * 100).round(2)

                col1, col2 = st.columns(2)

                with col1:
                    # Plot movement category growth
                    fig = px.pie(
                        move_summary,
                        values='predicted_sales',
                        names='movement_category',
                        title='Sales Distribution by Movement Category',
                        hole=0.4
                    )
                    st.plotly_chart(fig)

                with col2:
                    # Recommendation distribution
                    rec_counts = filtered_report['recommendation'].value_counts(
                    ).reset_index()
                    rec_counts.columns = ['recommendation', 'count']

                    fig = px.pie(
                        rec_counts,
                        values='count',
                        names='recommendation',
                        title='Recommendation Distribution',
                        hole=0.4
                    )
                    st.plotly_chart(fig)

                # Detailed recommendations table
                st.subheader("Detailed Recommendations")

                # Add sorting and filtering options
                sort_col, filter_col = st.columns(2)

                with sort_col:
                    sort_by = st.selectbox(
                        "Sort by",
                        options=["growth_pct", "predicted_sales",
                                 "inventory_risk", "category", "distributor_id"],
                        index=0
                    )

                with filter_col:
                    risk_filter = st.multiselect(
                        "Filter by risk level",
                        options=filtered_report['inventory_risk'].unique(),
                        default=filtered_report['inventory_risk'].unique()
                    )

                # Apply sorting and filtering
                display_df = filtered_report[filtered_report['inventory_risk'].isin(
                    risk_filter)]
                display_df = display_df.sort_values(
                    by=sort_by, ascending=False)

                # Select columns to display
                display_cols = ['distributor_id', 'sku', 'category', 'movement_category',
                                'predicted_sales', 'last_actual_sales', 'growth_pct',
                                'recommendation', 'inventory_risk', 'quarter_year']

                st.dataframe(display_df[display_cols])

                # Download button for full report
                csv = display_df.to_csv(index=False)
                st.download_button(
                    label="Download Full Report as CSV",
                    data=csv,
                    file_name="inventory_recommendations.csv",
                    mime="text/csv",
                )

                # Sales trend visualization for selected products
                st.subheader("Sales Trend Analysis")

                # Let user select specific products to view
                selected_skus = st.multiselect(
                    "Select products to visualize",
                    options=filtered_report['sku'].unique(),
                    default=filtered_report['sku'].unique()[:3] if len(
                        filtered_report['sku'].unique()) > 3 else filtered_report['sku'].unique()
                )

                if selected_skus:
                    # Get historical data for selected SKUs
                    historical_data = system.df[system.df['sku'].isin(
                        selected_skus)]

                    # Get future predictions for selected SKUs
                    future_data = filtered_report[filtered_report['sku'].isin(
                        selected_skus)]

                    # Prepare data for visualization
                    plot_data = []

                    # Add historical data
                    for _, row in historical_data.iterrows():
                        plot_data.append({
                            'sku': row['sku'],
                            'quarter_year': row['quarter_year'],
                            'sales': row['sales'],
                            'type': 'Historical'
                        })

                    # Add predicted data
                    for _, row in future_data.iterrows():
                        plot_data.append({
                            'sku': row['sku'],
                            'quarter_year': row['quarter_year'],
                            'sales': row['predicted_sales'],
                            'type': 'Predicted'
                        })

                    plot_df = pd.DataFrame(plot_data)

                    # Create time series plot
                    fig = px.line(
                        plot_df,
                        x='quarter_year',
                        y='sales',
                        color='sku',
                        line_dash='type',
                        markers=True,
                        title='Sales Trend with Predictions',
                        labels={'sales': 'Sales',
                                'quarter_year': 'Quarter-Year'}
                    )

                    st.plotly_chart(fig)

                    # Add festival impact analysis
                    st.subheader("Festival Impact Analysis")

                    # Create festival indicators
                    festival_cols = [
                        col for col in system.df.columns if col.startswith('is_')]

                    if festival_cols:
                        # Calculate average sales boost during festivals
                        festival_impact = []

                        for festival in festival_cols:
                            festival_name = festival.replace('is_', '').title()

                            # Calculate average sales during festival vs. non-festival periods
                            festival_sales = system.df[system.df[festival] == 1]['sales'].mean(
                            )
                            non_festival_sales = system.df[system.df[festival] == 0]['sales'].mean(
                            )

                            if pd.notnull(festival_sales) and pd.notnull(non_festival_sales) and non_festival_sales > 0:
                                boost_pct = (
                                    (festival_sales - non_festival_sales) / non_festival_sales * 100).round(2)

                                festival_impact.append({
                                    'Festival': festival_name,
                                    'Sales Boost %': boost_pct
                                })

                        if festival_impact:
                            impact_df = pd.DataFrame(festival_impact)

                            # Create festival impact visualization
                            fig = px.bar(
                                impact_df,
                                x='Festival',
                                y='Sales Boost %',
                                title='Average Sales Boost During Festivals',
                                color='Sales Boost %',
                                color_continuous_scale=['yellow', 'green'],
                            )

                            st.plotly_chart(fig)

                            # Provide actionable insights
                            st.subheader("Festival Planning Insights")

                            top_festivals = impact_df.sort_values(
                                'Sales Boost %', ascending=False)

                            if not top_festivals.empty:
                                top_festival = top_festivals.iloc[0]['Festival']
                                top_boost = top_festivals.iloc[0]['Sales Boost %']

                                st.write(
                                    f"💡 **Key Insight:** {top_festival} shows the highest sales boost at {top_boost}%.")
                                st.write(
                                    "Recommendations for festival planning:")
                                st.write(
                                    "1. Ensure adequate inventory for high-impact festivals")
                                st.write(
                                    f"2. Consider promotions and marketing campaigns around {top_festival}")
                                st.write(
                                    "3. Review historical festival performance to optimize stock levels")


# Entry point
if __name__ == "__main__":
    run_streamlit_app()

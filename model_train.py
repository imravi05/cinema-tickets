# train_model.py
import pandas as pd
from prophet import Prophet
import os
import pickle
import matplotlib.pyplot as plt

def train_and_save_model(file_path):
    # Load the dataset
    df = pd.read_csv("cinemaTicket_Ref.csv")

    # Convert the 'date' column to datetime objects
    df['date'] = pd.to_datetime(df['date'])

    # Aggregate 'total_sales' by 'date'
    daily_sales_df = df.groupby('date')['total_sales'].sum().reset_index()

    # Rename columns to 'ds' and 'y' for Prophet
    daily_sales_df.rename(columns={'date': 'ds', 'total_sales': 'y'}, inplace=True)

    # Initialize the Prophet model
    model = Prophet(
        daily_seasonality=True,
        # You can add more parameters here if needed, e.g., seasonality_mode='multiplicative'
    )

    # Fit the model to the data
    model.fit(daily_sales_df)

    # Create a directory to save the model if it doesn't exist
    output_dir = "model"
    os.makedirs(output_dir, exist_ok=True)

    # Save the trained model
    model_path = os.path.join(output_dir, 'prophet_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    print(f"Prophet model trained and saved to {model_path}")

    # Optional: Generate a forecast and plot for verification
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    fig1 = model.plot(forecast)
    plt.title('Prophet Forecast for Total Sales (Training Verification)')
    plt.xlabel('Date')
    plt.ylabel('Total Sales')
    plt.savefig(os.path.join(output_dir, 'forecast_plot.png')) # Save plot
    plt.show()

    fig2 = model.plot_components(forecast)
    plt.savefig(os.path.join(output_dir, 'components_plot.png')) # Save plot
    plt.show()


if __name__ == '__main__':
    # Assuming 'cinemaTicket_Ref.csv' is in the same directory as train_model.py
    train_and_save_model('cinemaTicket_Ref.csv')
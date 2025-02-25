import datetime
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

## ## ## ## #
## GLOBALS ##
## ## ## ## #

# known failure anomalies (derived from PDF with failure and dataset information)
# linked in this project
# they are a range of timestamps
known_failure_anomalies = [
    ("4/18/2020 0:00", "4/18/2020 23:59"),
    ("5/29/2020 23:30", "5/30/2020 6:00"),
    ("6/5/2020 10:00", "6/7/2020 14:30"),
    ("7/15/2020 14:30", "7/15/2020 19:00"),
]

time_format = "%m/%d/%Y %H:%M"


## ## ## ## #
## METHODS ##
## ## ## ## #


# peeking at the data to see what we are working with
def peeking_at_data(df):
    # head and tail
    print("- - - - DF HEAD - - - - ")
    print(df.head())
    print("\n - - - - DF TAIL - - - - ")
    print(df.tail())

    # shape and info
    print("\n - - - - DF SHAPE - - - - ")
    print(df.shape)
    print("\n - - - - DF INFO - - - - ")
    print(df.info())
    print("\n - - - - DF DESCRIBE - - - - ")
    print(df.describe())

    # types and null values (if any)
    print("\n - - - - DATA TYPES - - - - ")
    print(df.dtypes)
    print("\n - - - - NULL VALUES - - - - ")
    print(df.isnull().sum())


# helper method to convert known points of failure to datetime format
def convert_to_datetime_helper(datestring, format=time_format):
    return datetime.datetime.strptime(datestring, format)


# takes the globally defined failure anomalies and makes regular dateTime format
def convert_failure_anomalies_to_datetime():
    known_failer_anomalies_datetimes = [
        tuple(map(convert_to_datetime_helper, x)) for x in known_failure_anomalies
    ]
    return known_failer_anomalies_datetimes


# draw some varying plots / heatmaps to see if we can make anything of inital raw data
def post_data_clearning_visualization(df):
    print("Writing first visualization.....")
    plt.style.use("ggplot")  # dark_background aint too shabby either
    df.plot(
        subplots=True,
        layout=(6, 3),
        figsize=(22, 22),
        fontsize=10,
        linewidth=1,
        sharex=False,
        title="Visualization of the original Time Series",
    )
    plt.savefig("visualization_of_original_time_series.png")
    plt.clf()
    print("......finished")
    print("Writing second visualization.....")
    plt.style.use("ggplot")
    df.plot(
        kind="kde",
        subplots=True,
        layout=(6, 3),
        figsize=(22, 22),
        fontsize=10,
        linewidth=2,
        sharex=False,
        title="Second visualization of the original Time Series",
    )
    plt.savefig("kde_visualization_of_original_time_series.png")
    plt.clf()
    print("......finished")
    print("Writing correlation matrix.....")

    # compute the correlation matrix
    corr_matrix = df.corr(method="spearman")

    # set the figure size
    plt.figure(figsize=(12, 10))

    # generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        annot_kws={"size": 12},
        cmap="coolwarm",
    )
    plt.title("Correlation Matrix of the Features")
    plt.savefig("correlation_matrix.png")
    plt.clf()
    print("......finished")


def concatenating_failure_flag(df, known_failure_anomalies):
    df["failure_anomaly"] = [0] * df.shape[0]
    df["timestamp"] = pd.to_datetime(df["timestamp"], format=time_format)

    # Add a column to mark anomaly periods
    for start, end in known_failure_anomalies:
        df.loc[
            (df["timestamp"] >= start) & (df["timestamp"] <= end), "failure_anomaly"
        ] = 1
    return df


# after a look at the data and some of the literature linked in the PDF that came
# with the dataset, it is apparent that the time sampling fluctuates
# the measurements were taken at 1Mhz (which is every 10 seconds in this case),
# but seem to flucuate. This method will look for and isolate these fluctuations
def time_fluctuation_(df):
    # i had to do this because pandas was getting pissy and throwing errors at me
    new_df = df.copy()

    # preserve the og indx
    new_df["original_index"] = new_df.index

    # calculate time difference between rows
    new_df["time_diff"] = new_df["timestamp"].diff().dt.total_seconds()

    # identify where time difference exceeds 10 seconds
    breaks_df = new_df[new_df["time_diff"] > 10].copy()

    # metadata
    breaks_df["break_duration"] = breaks_df["time_diff"]  # duration of each break
    breaks_df["prev_timestamp"] = new_df["timestamp"].shift(1)  # add previous timestamp

    # drop unnecessary col
    breaks_df = breaks_df[
        [
            "original_index",
            "prev_timestamp",
            "timestamp",
            "break_duration",
            "failure_anomaly",
        ]
    ]

    # reset index for clean output
    breaks_df = breaks_df.reset_index(drop=True)
    print(f"Number of breaks: {len(breaks_df)}")
    print("Breaks details:")
    print(breaks_df)
    breaks_df.to_csv("breaks_in_time.csv", sep=",")


def target_features_visualisation(df):

    key_sensors = [
        "Motor_current",
        "Oil_temperature",
        "DV_eletric",
        "TP2",
        "DV_pressure",
    ]
    for sensor in key_sensors:
        plt.figure(figsize=(12, 6))
        plt.plot(
            df["timestamp"],
            df[sensor],
            label=f"{sensor} trends",
            color="blue",
        )

        # Overlay failure events
        failure_times = df[df["failure_anomaly"] == 1]["timestamp"]

        # Get current y-axis limits to plot vertical lines correctly
        ymin, ymax = plt.ylim()

        # Plot all failure events at once and add label only once
        plt.vlines(
            x=failure_times,
            ymin=ymin,
            ymax=ymax,
            colors="red",
            linestyles="--",
            alpha=0.7,
            label="Failure Event",
        )

        plt.xlabel("Time")
        plt.ylabel(sensor)
        plt.title(f"{sensor} plotted with failure events")

        # Place the legend outside the plot area on the right
        plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.grid()

        # Adjust layout to make room for the legend
        plt.tight_layout()
        plt.savefig(f"{sensor}_failure.png", bbox_inches="tight")
        plt.clf()


## ## ## #
## MAIN ##
## ## ## #
if __name__ == "__main__":
    # remove comment to write to an output file
    sys.stdout = open("data_cleaning_output", "w")

    df = pd.read_csv("../../../metropt+3+dataset/MetroPT3(AirCompressor).csv")

    # dropping the "unnamed" column from data - is not useful
    df = df.drop(columns=["Unnamed: 0"])

    # initial look at dataset
    peeking_at_data(df)

    # convert the 'timestamp' column to datetime type
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # check the data types again to confirm
    print("- - - - POST CHANGING TYPE OF TIMESTAMP - - - - ")
    print(df.dtypes)

    # data wrangling
    # getting the known failure windows into usable format
    known_failure_anomaly_datetimes = convert_failure_anomalies_to_datetime()

    # print out if needed
    print("Failure datetimes")
    print(known_failure_anomaly_datetimes)

    # adding the failure anomalies as new col
    df_with_failures = concatenating_failure_flag(df, known_failure_anomaly_datetimes)

    # debugging
    print("Data after adding anomalies")
    print(df)

    # filtering out the noise - >10 sec variations
    print(" - - - - - - time fluncuations - - - - - - ")
    print()
    time_fluctuation_(df_with_failures)

    # removing any of the datapoints that failed to stay in the 1Mhz range (10 sec variation)
    breaks_in_time_df = pd.read_csv("breaks_in_time.csv")
    indices_to_remove = breaks_in_time_df["original_index"].tolist()

    # Drop rows that match
    df_with_failures = df_with_failures.drop(index=indices_to_remove)

    # noise is now removed
    print(df_with_failures)

    # splitting the data into 2 unique sets of failure / non failure
    just_failures_df = df_with_failures[df_with_failures["failure_anomaly"] == 1]
    no_failures_df = df_with_failures[df_with_failures["failure_anomaly"] == 0]

    # save all the new formatted datasets for future use
    df_with_failures.to_csv("data_with_failures_included.csv", sep=",")
    just_failures_df.to_csv("just_failures_df.csv", sep=",")
    no_failures_df.to_csv("no_failures_df.csv", sep=",")

    # visualization post data cleansing/noise removal
    df_no_time = df_with_failures.drop(columns="timestamp")
    post_data_clearning_visualization(df_no_time)

    target_features_visualisation(df_with_failures)

    # remove comment to write to an output file
    sys.stdout.close()

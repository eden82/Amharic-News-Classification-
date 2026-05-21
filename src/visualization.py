import matplotlib.pyplot as plt

def plot_category_distribution(df, column="category"):
    """Plot the distribution of categories in the dataset."""
    # Use full Unicode font that supports Amharic + English
    plt.rcParams["font.family"] = "Noto Sans"

    # Plot category distribution
    df[column].value_counts().plot(kind="bar")

    # Titles and labels
    plt.title("Category Distribution")
    plt.xlabel("Category")
    plt.ylabel("Count")

    # Rotate labels for readability
    plt.xticks(rotation=45)

    # Show plot
    plt.show()

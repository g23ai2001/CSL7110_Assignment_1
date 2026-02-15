from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, regexp_extract, explode, collect_list, size, count, desc, lower, trim, min, max,avg
from pyspark.sql.types import StringType, ArrayType, StructType, StructField, IntegerType
import re

print("Initializing Spark Session...")
spark = SparkSession.builder \
    .appName("Q12: Author Influence Network") \
    .master("local[*]") \
    .config("spark.driver.memory", "8g") \
    .config("spark.executor.memory", "8g") \
    .config("spark.sql.shuffle.partitions", "10") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

print("Spark Session created!\n")


print("="*80)
print("STEP 1: Loading Books and Extracting Metadata")
print("="*80)

books_df = spark.read.text("dataset/*.txt", wholetext=True) \
    .selectExpr("input_file_name() as file_path", "value as text")

books_df = books_df.withColumn(
    "file_name", 
    regexp_extract(col("file_path"), r"([^/\\]+\.txt)$", 1)
)

total_books = books_df.count()
print(f"Total books loaded: {total_books}\n")



print("="*80)
print("STEP 2: Extracting Author and Release Year")
print("="*80)

def extract_author(text):
    
    if not text:
        return None
    
    # Search first 2000 characters for author information
    header = text[:2000]
    
    # Pattern 1: "Author: Name"
    patterns = [
        r"Author:\s*(.+?)(?:\r?\n)",
        r"Author:\s*(.+?)(?:\[|$)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, header, re.IGNORECASE)
        if match:
            author = match.group(1).strip()
            # Clean up common suffixes
            author = re.sub(r'\s*\[.*?\]\s*', '', author)  # Remove [Translator], [Editor], etc.
            author = author.strip()
            if author and len(author) > 2:  # Avoid very short matches
                return author
    
    return None

def extract_release_year(text):
  
    if not text:
        return None
    
    header = text[:2000]
    
    # First try to get from "Release Date" field
    date_pattern = r"Release Date:\s*(.+?)(?:\[|$|\r?\n\r?\n)"
    match = re.search(date_pattern, header, re.IGNORECASE)
    
    if match:
        date_str = match.group(1)
        # Extract 4-digit year
        year_match = re.search(r'\b(19\d{2}|20\d{2})\b', date_str)
        if year_match:
            return int(year_match.group(1))
    
    return None

# Register UDFs
extract_author_udf = udf(extract_author, StringType())
extract_release_year_udf = udf(extract_release_year, IntegerType())

# Extract metadata
books_with_metadata = books_df \
    .withColumn("author", extract_author_udf(col("text"))) \
    .withColumn("release_year", extract_release_year_udf(col("text"))) \
    .select("file_name", "author", "release_year")

# Filter books with both author and year
books_complete = books_with_metadata.filter(
    (col("author").isNotNull()) & (col("release_year").isNotNull())
)

books_complete.cache()
complete_count = books_complete.count()

print(f"\nBooks with complete metadata (author + year): {complete_count}")
print(f"Books without complete metadata: {total_books - complete_count}\n")

print("Sample of extracted metadata:")
books_complete.orderBy("file_name").show(20, truncate=60)


print("\n" + "="*80)
print("Author Statistics")
print("="*80)

author_book_counts = books_complete.groupBy("author") \
    .agg(count("*").alias("book_count")) \
    .orderBy(desc("book_count"))

print("\nTop 20 Authors by Number of Books:")
author_book_counts.show(20, truncate=60)

unique_authors = author_book_counts.count()
print(f"\nTotal unique authors: {unique_authors}")


print("\n" + "="*80)
print("STEP 4: Defining Author Influence Relationships")
print("="*80)



# Self-join to create author pairs
# Rename columns for clarity
authors_early = books_complete.select(
    col("author").alias("author_early"),
    col("release_year").alias("year_early"),
    col("file_name").alias("book_early")
)

authors_late = books_complete.select(
    col("author").alias("author_late"),
    col("release_year").alias("year_late"),
    col("file_name").alias("book_late")
)

# Cross join to create all pairs
author_pairs = authors_early.crossJoin(authors_late)

# Define influence criteria
INFLUENCE_WINDOW_MIN = 1   # Minimum years gap
INFLUENCE_WINDOW_MAX = 20  # Maximum years gap

influence_edges = author_pairs.filter(
    # Different authors
    (col("author_early") != col("author_late")) &
    # Early author published before late author
    (col("year_early") < col("year_late")) &
    # Within influence window
    ((col("year_late") - col("year_early")) >= INFLUENCE_WINDOW_MIN) &
    ((col("year_late") - col("year_early")) <= INFLUENCE_WINDOW_MAX)
).select(
    col("author_early").alias("influencer"),
    col("author_late").alias("influenced"),
    col("year_early"),
    col("year_late"),
    (col("year_late") - col("year_early")).alias("year_gap")
)

print(f"\nInfluence window: {INFLUENCE_WINDOW_MIN} to {INFLUENCE_WINDOW_MAX} years")
print("\nCalculating influence relationships...")

influence_edges.cache()
total_edges = influence_edges.count()

print(f"Total influence relationships found: {total_edges}")

print("\nSample influence relationships:")
influence_edges.orderBy("year_gap").show(20, truncate=60)


print("\n" + "="*80)
print("STEP 5: Creating Author-Level Influence Network")
print("="*80)


author_influence = influence_edges.groupBy("influencer", "influenced") \
    .agg(
        count("*").alias("num_connections"),  # Number of book pairs
        min("year_gap").alias("min_year_gap"),
        max("year_gap").alias("max_year_gap")
    )

author_influence.cache()
unique_edges = author_influence.count()

print(f"Unique author-to-author influence relationships: {unique_edges}")

print("\nSample of author influence network:")
author_influence.orderBy(desc("num_connections")).show(20, truncate=60)


print("\n" + "="*80)
print("STEP 6: Calculating Network Metrics (In-Degree and Out-Degree)")
print("="*80)



# Calculate OUT-DEGREE 
out_degree = author_influence.groupBy("influencer") \
    .agg(count("influenced").alias("out_degree")) \
    .withColumnRenamed("influencer", "author")

print("\nCalculating out-degree...")
out_degree.cache()
print(f"Out-degree calculated for {out_degree.count()} authors")

# Calculate IN DEGREE 
in_degree = author_influence.groupBy("influenced") \
    .agg(count("influencer").alias("in_degree")) \
    .withColumnRenamed("influenced", "author")

print("Calculating in-degree...")
in_degree.cache()
print(f"In-degree calculated for {in_degree.count()} authors")

# Combine both metrics
author_metrics = out_degree.join(in_degree, on="author", how="outer").fillna(0)

author_metrics.cache()
total_authors_in_network = author_metrics.count()

print(f"\n Total authors in influence network: {total_authors_in_network}")


print("\n" + "="*80)
print("STEP 7: Identifying Top Influential Authors")
print("="*80)

print("\n" + "="*60)
print("TOP 5 AUTHORS BY OUT-DEGREE (Most Influential)")
print("="*60)
print("These authors influenced the most other authors:\n")

top_out_degree = author_metrics.orderBy(desc("out_degree")).limit(5)
top_out_results = top_out_degree.collect()

for i, row in enumerate(top_out_results, 1):
    print(f"{i}. {row['author']:<40} Out-Degree: {row['out_degree']:<5} In-Degree: {row['in_degree']}")

print("\n" + "="*60)
print("TOP 5 AUTHORS BY IN-DEGREE (Most Influenced)")
print("="*60)
print("These authors were influenced by the most other authors:\n")

top_in_degree = author_metrics.orderBy(desc("in_degree")).limit(5)
top_in_results = top_in_degree.collect()

for i, row in enumerate(top_in_results, 1):
    print(f"{i}. {row['author']:<40} In-Degree: {row['in_degree']:<5} Out-Degree: {row['out_degree']}")



print("\n" + "="*80)
print("Saving Results...")
print("="*80)

# Save influence edges
author_influence.coalesce(1).write.csv(
    "output/author_influence_network.csv",
    header=True,
    mode="overwrite"
)
print("Influence network saved to: output/author_influence_network.csv")


author_metrics.coalesce(1).write.csv(
    "output/author_metrics.csv",
    header=True,
    mode="overwrite"
)
print("Author metrics saved to: output/author_metrics.csv")

# Save top authors
top_out_degree.union(top_in_degree).distinct().coalesce(1).write.csv(
    "output/top_authors.csv",
    header=True,
    mode="overwrite"
)
print("Top authors saved to: output/top_authors.csv")

spark.stop()
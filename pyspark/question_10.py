from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, length, count, avg, desc
from pyspark.sql.types import StringType, IntegerType
import re

print("Initializing Spark Session...")
spark = SparkSession.builder \
    .appName("Q10: Book Metadata Extraction") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "10") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

# Define extraction functions
def extract_title(text):
    if not text: return None
    pattern = r"Title:\s*(.+?)(?:\r?\n)"
    match = re.search(pattern, text[:2000], re.IGNORECASE)
    return match.group(1).strip() if match else None

def extract_release_date(text):
    if not text: return None
    pattern = r"Release Date:\s*(.+?)(?:\[|$|\r?\n\r?\n)"
    match = re.search(pattern, text[:2000], re.IGNORECASE)
    return match.group(1).strip() if match else None

def extract_language(text):
    if not text: return None
    pattern = r"Language:\s*(.+?)(?:\r?\n)"
    match = re.search(pattern, text[:2000], re.IGNORECASE)
    return match.group(1).strip() if match else None

def extract_encoding(text):
    if not text: return None
    pattern = r"Character set encoding:\s*(.+?)(?:\r?\n)"
    match = re.search(pattern, text[:2000], re.IGNORECASE)
    return match.group(1).strip() if match else None

def extract_year(date_str):
    if not date_str: return None
    match = re.search(r'\b(19\d{2}|20\d{2})\b', date_str)
    return int(match.group(1)) if match else None

# Register UDFs
extract_title_udf = udf(extract_title, StringType())
extract_release_date_udf = udf(extract_release_date, StringType())
extract_language_udf = udf(extract_language, StringType())
extract_encoding_udf = udf(extract_encoding, StringType())
extract_year_udf = udf(extract_year, IntegerType())

# Load books
print("\nLoading books from dataset/*.txt...")
books_df = spark.read.text("dataset/*.txt", wholetext=True) \
    .selectExpr("input_file_name() as file_path", "value as text")

from pyspark.sql.functions import regexp_extract
books_df = books_df.withColumn(
    "file_name", 
    regexp_extract(col("file_path"), r"([^/\\]+\.txt)$", 1)
)

print(f"Total books loaded: {books_df.count()}\n")

# Extract metadata - DO NOT CACHE to save memory
print("Extracting metadata...")
books_with_metadata = books_df \
    .withColumn("title", extract_title_udf(col("text"))) \
    .withColumn("release_date", extract_release_date_udf(col("text"))) \
    .withColumn("language", extract_language_udf(col("text"))) \
    .withColumn("encoding", extract_encoding_udf(col("text"))) \
    .withColumn("release_year", extract_year_udf(col("release_date")))

# Drop the full text column to save memory
books_metadata_only = books_with_metadata.select(
    "file_name", "title", "release_date", "release_year", "language", "encoding"
)

# Cache only the small metadata (not the full text)
books_metadata_only.cache()
books_metadata_only.count()  # Force cache

print("\nSample of extracted metadata:")
books_metadata_only.show(10, truncate=50)

# ANALYSIS 1: Books per year
print("\n" + "="*70)
print("ANALYSIS 1: Number of Books Released Each Year")
print("="*70)
books_per_year = books_metadata_only \
    .filter(col("release_year").isNotNull()) \
    .groupBy("release_year") \
    .agg(count("*").alias("book_count")) \
    .orderBy("release_year")
books_per_year.show(30)
books_per_year.coalesce(1).write.csv("output/books_per_year.csv", header=True, mode="overwrite")
print("Results saved to: output/books_per_year.csv\n")

# ANALYSIS 2: Most common language
print("\n" + "="*70)
print("ANALYSIS 2: Most Common Language in Dataset")
print("="*70)
language_counts = books_metadata_only \
    .filter(col("language").isNotNull()) \
    .groupBy("language") \
    .agg(count("*").alias("count")) \
    .orderBy(desc("count"))
language_counts.show(10)

most_common = language_counts.first()
if most_common:
    print(f"\n>>> Most common language: {most_common['language']} ({most_common['count']} books)")

language_counts.coalesce(1).write.csv("output/language_distribution.csv", header=True, mode="overwrite")
print("Results saved to: output/language_distribution.csv\n")

# ANALYSIS 3: Average title length
print("\n" + "="*70)
print("ANALYSIS 3: Average Length of Book Titles")
print("="*70)
books_with_title_length = books_metadata_only \
    .filter(col("title").isNotNull()) \
    .withColumn("title_length", length(col("title")))

avg_length = books_with_title_length.agg(avg("title_length")).first()[0]
print(f"\n>>> Average title length: {avg_length:.2f} characters\n")

print("Longest titles:")
books_with_title_length.select("file_name", "title", "title_length") \
    .orderBy(desc("title_length")).show(5, truncate=60)

print("\nShortest titles:")
books_with_title_length.select("file_name", "title", "title_length") \
    .orderBy("title_length").show(5, truncate=60)

books_with_title_length.select("file_name", "title", "title_length") \
    .coalesce(1).write.csv("output/title_lengths.csv", header=True, mode="overwrite")
print("Results saved to: output/title_lengths.csv\n")

# Data Quality Report
print("\n" + "="*70)
print("DATA QUALITY REPORT")
print("="*70)
total_books = books_metadata_only.count()
title_count = books_metadata_only.filter(col("title").isNotNull()).count()
date_count = books_metadata_only.filter(col("release_date").isNotNull()).count()
year_count = books_metadata_only.filter(col("release_year").isNotNull()).count()
language_count = books_metadata_only.filter(col("language").isNotNull()).count()
encoding_count = books_metadata_only.filter(col("encoding").isNotNull()).count()

print(f"Total books: {total_books}")
print(f"Books with title: {title_count} ({title_count/total_books*100:.1f}%)")
print(f"Books with release date: {date_count} ({date_count/total_books*100:.1f}%)")
print(f"Books with release year: {year_count} ({year_count/total_books*100:.1f}%)")
print(f"Books with language: {language_count} ({language_count/total_books*100:.1f}%)")
print(f"Books with encoding: {encoding_count} ({encoding_count/total_books*100:.1f}%)")

# Save complete metadata
print("\n" + "="*70)
print("Saving complete metadata...")
print("="*70)
books_metadata_only.coalesce(1).write.csv("output/complete_book_metadata.csv", header=True, mode="overwrite")
print("✓ Complete metadata saved to: output/complete_book_metadata.csv")

print("\n" + "="*70)
print("Question 10 Complete!")
print("="*70)

spark.stop()
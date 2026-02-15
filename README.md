
## 📋 Assignment Overview

This repository contains complete implementations for a comprehensive Big Data processing assignment covering **Hadoop MapReduce** and **Apache Spark**. The project analyzes the **Project Gutenberg dataset** (425 books) using distributed computing frameworks.

**GitHub Repository:** [https://github.com/g23ai2001/CSL7110_Assignment_1](https://github.com/g23ai2001/CSL7110_Assignment_1)

## 🎯 Assignment Components

### Section A: Hadoop MapReduce (Questions 1-9)

| Question | Topic | Description |
|----------|-------|-------------|
| Q1 | WordCount Setup | Basic WordCount implementation and verification |
| Q2 | Map Phase Analysis | Output pairs and data type identification |
| Q3 | Reduce Phase Analysis | Input pairs and data type identification |
| Q4 | Map Class Implementation | Mapper class definition with proper types |
| Q5 | Text Processing | Map function with punctuation handling |
| Q6 | Reducer Implementation | Reduce function for word counting |
| Q7 | Execution Verification | Running WordCount on 200.txt file |
| Q8 | HDFS Concepts | Replication factor analysis for files vs directories |
| Q9 | Performance Analysis | Split size experimentation (8MB, 4MB, 1MB) |

### Section B: Apache Spark (Questions 10-12)

| Question | Topic | Description |
|----------|-------|-------------|
| Q10 | Metadata Extraction | Regex-based extraction and statistical analysis |
| Q11 | TF-IDF Similarity | Document vectorization and cosine similarity |
| Q12 | Author Network | Influence network construction and graph metrics |

## 🛠️ Technologies & Tools

- **Hadoop 3.4.1** - Distributed storage and MapReduce processing
- **Apache Spark 3.5.0** - Large-scale data processing engine
- **Python 3.x** - Primary language for Spark applications
- **PySpark** - Python API for Apache Spark
- **Java** - For MapReduce implementations

## 📦 Dataset

- **Source:** Project Gutenberg (D184MB dataset)
- **Format:** Text files (.txt)
- **Size:** 425 books
- **Content:** Public domain literature with metadata headers

## 🚀 Installation & Setup

### Prerequisites
```bash
# Java (required for both Hadoop and Spark)
java -version  # Should be Java 11 or higher

# Hadoop installation
hadoop version  # Should show Hadoop 3.4.1

# Spark installation
spark-submit --version  # Should show Spark 3.5.0
```

### Python Dependencies
```bash
pip install -r requirements.txt
```

**requirements.txt contents:**
```
pyspark==3.5.0
numpy>=1.21.0
pandas>=1.3.0
```

### Dataset Setup
```bash
# Download and unzip the Project Gutenberg dataset
unzip D184MB.zip

# Copy to HDFS (for Hadoop questions)
hdfs dfs -mkdir -p /user/$USER/input
hdfs dfs -copyFromLocal 200.txt /user/$USER/input/
```

## 📂 Repository Structure
```
CSL7110_Assignment_1/
│
├── Hadoop/                    # Hadoop MapReduce implementations
│   ├── WordCount.java           # Q1, Q4, Q5, Q6, Q7
                   
│
├── Spark/                        # Apache Spark implementations
│   ├── question_10.py   # Q10
│   ├── question_11.py      # Q11
│   └── question_12.py        # Q12
│
├── output/                       # Execution results and screenshots
│   ├── q1_wordcount_output.txt
│   ├── q7_execution_screenshot.png
│   ├── q9_performance_comparison.csv
│   ├── q10_metadata_analysis.csv
│   ├── q11_similarity_results.csv
│   └── q12_network_metrics.csv
│
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🔧 How to Run

### Hadoop MapReduce Questions

#### Q1-Q7: WordCount Implementation
```bash
# Compile Java code
javac -classpath $HADOOP_HOME/share/hadoop/mapreduce/hadoop-mapreduce-examples-3.4.1.jar WordCount.java
jar -cvf WordCount.jar *.class

# Run WordCount
hadoop jar WordCount.jar WordCount /user/$USER/input/200.txt /user/$USER/output

# View results
hdfs dfs -cat /user/$USER/output/part-r-00000
```

#### Q9: Split Size Analysis
```bash
# Run with different split sizes
hadoop jar WordCount.jar WordCount /user/$USER/input/200.txt /user/$USER/output1  # Default (8MB)
hadoop jar WordCount.jar WordCount /user/$USER/input/200.txt /user/$USER/output2  # 4MB
hadoop jar WordCount.jar WordCount /user/$USER/input/200.txt /user/$USER/output3  # 1MB
```



## 📊 Key Results & Findings

### Q9: MapReduce Performance Analysis

| Split Size | Execution Time | Mappers | Performance Change |
|------------|---------------|---------|-------------------|
| 8 MB (default) | 18,834 ms | 1 | Baseline |
| 4 MB | 19,951 ms | 2 | +6% slower |
| 1 MB | 48,926 ms | 8 | +160% slower |

**Finding:** Smaller splits increase overhead due to task scheduling and container initialization.

### Q10: Metadata Extraction Results

- **Total Books Analyzed:** 425
- **Books with Complete Metadata:** 373 (87.85%)
- **Most Common Language:** English (403 books)
- **Average Title Length:** 32.85 characters
- **Year Range:** 1970-2008

**Regular Expression Patterns:**
```python
Title:     r"Title:\s*(.+?)(?:\r?\n)"
Date:      r"Release Date:\s*(.+?)(?:\[|$|\r?\n\r?\n)"
Year:      r'\b(19\d{2}|20\d{2})\b'
Language:  r"Language:\s*(.+?)(?:\r?\n)"
Encoding:  r"Character set encoding:\s*(.+?)(?:\r?\n)"
```

### Q11: TF-IDF Document Similarity

**Top 5 Most Distinctive Words for Book "18.txt":**
1. unto (TF-IDF: 0.82)
2. israel (TF-IDF: 0.81)
3. thou (TF-IDF: 0.79)
4. thy (TF-IDF: 0.87)
5. thee (TF-IDF: 0.84)

**Cosine Similarity:** Successfully computed 90,100 pairwise comparisons (425 books).

### Q12: Author Influence Network

**Network Statistics:**
- **Total Authors:** 212
- **Total Influence Edges:** 21,646
- **Time Window:** 1-20 years

**Top 5 Most Influential Authors (Highest Out-Degree):**
1. Robert Louis Stevenson (Out-Degree: 286)
2. Charles Dodgson (Lewis Carroll) (Out-Degree: 285)
3. Herman Melville (Out-Degree: 285)
4. Brendan P. Kehoe (Out-Degree: 282)
5. Edgar Rice Burroughs (Out-Degree: 161)

**Top 5 Most Influenced Authors (Highest In-Degree):**
1. Anonymous (In-Degree: 218)
2. Jerome K. Jerome (In-Degree: 246)
3. Dante Alighieri (In-Degree: 237)
4. Robert Louis Stevenson (In-Degree: 205)
5. Ludwig van Beethoven (In-Degree: 265)

## 💡 Technical Highlights

### Hadoop MapReduce Insights

1. **Data Types:** Used Hadoop's `LongWritable`, `Text`, and `IntWritable` for type safety
2. **Text Processing:** Implemented regex-based punctuation removal: `line.replaceAll("[^a-z0-9\\s]", "")`
3. **Performance:** Demonstrated that optimal split size balances parallelism with scheduling overhead

### Apache Spark Insights

1. **Regex Patterns:** Handled multiple date formats and encoding variations
2. **TF-IDF:** Implemented complete pipeline from tokenization to similarity scoring
3. **Network Analysis:** Used DataFrame operations for efficient degree calculations
4. **Scalability:** Leveraged Spark's lazy evaluation and in-memory processing


## 🎓 Learning Outcomes

- ✅ Practical experience with Hadoop MapReduce framework
- ✅ Understanding of HDFS architecture and replication
- ✅ Proficiency in PySpark DataFrame and RDD operations
- ✅ Regular expression pattern matching for semi-structured text
- ✅ TF-IDF implementation and document similarity analysis
- ✅ Graph network construction and analysis with Spark
- ✅ Performance tuning and optimization strategies

## 📝 Assignment Questions Answered

All 12 questions have been comprehensively addressed:
- **Q1-Q9:** Hadoop MapReduce implementation with performance analysis
- **Q10:** Metadata extraction with regex explanation and data quality assessment
- **Q11:** TF-IDF calculation with similarity analysis and scalability discussion
- **Q12:** Author influence network with degree metrics and implementation justification

## 🔗 References

- [Apache Hadoop Documentation](https://hadoop.apache.org/docs/stable/)
- [Apache Spark Documentation](https://spark.apache.org/docs/latest/)
- [Project Gutenberg](https://www.gutenberg.org/)
- [HDFS Commands Reference](https://hadoop.apache.org/docs/stable/hadoop-project-dist/hadoop-common/FileSystemShell.html)

## 📧 Contact

For questions regarding this submission:
- **Student ID:** M25DE1001



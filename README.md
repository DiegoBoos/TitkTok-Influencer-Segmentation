# TikTok Influencer Segmentation

## Project Overview

This project applies machine learning techniques to segment TikTok influencers based on their engagement metrics, follower counts, and content production patterns. By identifying distinct influencer profiles, marketers and content strategists can develop more targeted approaches for influencer marketing campaigns.

## Business Context

The rise of social media influencers has transformed digital marketing, with TikTok emerging as one of the most influential platforms. However, not all influencers deliver the same value or engagement. This segmentation analysis helps identify which types of influencers might be most effective for different marketing goals.

## Data Description

The dataset (`profiles_dataset.csv`) contains information about 1,000 TikTok influencer profiles with the following key features:

- **Engagement metrics**: Average engagement rate, comment engagement rate, like engagement rate
- **Audience size**: Followers count
- **Activity metrics**: Following count, likes received, videos count
- **Account status**: Verification status
- **Profile information**: Biography, account ID, nickname

## Methodology

The analysis follows these key steps:

1. **Data Cleaning**
   - Handling missing values and duplicates
   - Converting boolean values to integers for analysis

2. **Feature Selection**
   - Focusing on engagement metrics, follower counts, and content production

3. **Outlier Handling**
   - Using IQR (Interquartile Range) method to remove extreme values in engagement metrics
   - Preserving natural distribution of follower counts

4. **Feature Scaling**
   - Standardizing features to ensure fair contribution in the clustering algorithm

5. **Cluster Analysis**
   - Determining optimal number of clusters using the Elbow Method
   - Applying K-means clustering algorithm
   - Visualizing clusters and interpreting results

## Key Findings

The analysis identified 5 distinct influencer segments:

1. **Mid-Tier Mainstream Influencers** (Cluster 0)
   - Moderate followers (106K average)
   - Low engagement rates (1.8%)
   - Consistent content production

2. **Celebrity Accounts** (Cluster 1)
   - Massive follower base (12.5M average)
   - Very low engagement rates (0.95%)
   - High content output (1590 videos)
   - 50% verified accounts

3. **Niche Engagement Specialists** (Cluster 2)
   - Moderate follower count (70K average)
   - High engagement rates (10.2%)
   - Focused content strategy (155 videos)

4. **Community Builders** (Cluster 3)
   - Smaller audience (23K followers)
   - Moderate engagement (1.9%)
   - High following count (7482 average)
   - Active content creation (580 videos)

5. **Verified Content Creators** (Cluster 4)
   - Large follower base (1.45M average)
   - Moderate engagement rates (3.1%)
   - 100% verified accounts
   - Substantial content library

## Running the Code

1. Clone the repository
2. Install required packages: `pip install -r requirements.txt`
# **Project Overview**
This project applies **unsupervised learning techniques** to segment TikTok influencers based on their engagement metrics, follower counts, and content production patterns.

In a real-world scenario, identifying distinct influencer profiles helps marketers and content strategists develop **targeted influencer marketing campaigns**.

## **📊 Data Description**
The dataset includes **1,000 TikTok influencer profiles** with features such as:
- **Engagement Metrics**: Like & comment rates, average engagement rate
- **Audience Size**: Followers count
- **Activity Metrics**: Number of posts, likes received
- **Verification Status**: Account authentication details

## **🛠 Methodology**
### **1️⃣ Data Cleaning**
- Removing missing values and duplicates
- Standardizing categorical values

### **2️⃣ Feature Selection & Scaling**
- Focusing on engagement, content metrics & follower counts
- Standardizing numerical features for fair weight distribution

### **3️⃣ Clustering Analysis**
- Determining optimal clusters via **Elbow Method**
- Implementing **K-means clustering** for segmentation

## **📌 Key Findings**
This analysis identified **five influencer segments:**
1. **Mid-Tier Mainstream Influencers** – Moderate followers (106K avg.), low engagement (1.8%)
2. **Celebrity Accounts** – Large followings (12.5M avg.), very low engagement (0.95%)
3. **Niche Engagement Specialists** – Small but highly engaged audience (10.2%)
4. **Community Builders** – Smaller following (23K avg.), moderate engagement
5. **Verified Content Creators** – Verified, large audiences (1.45M avg.), moderate engagement

## **📓 Full Report**
[Notebook Report](https://github.com/DiegoBoos/TitkTok-Influencer-Segmentation/blob/main/docs/influencer_segmentation.pdf)

## **💻 Running the Code**
1. **Clone the repository**
2. **Set up a virtual environment**
   ```sh
   python -m venv venv
   source venv/bin/activate  # Mac/Linux
   venv\Scripts\activate  # Windows
   pip install -r requirements.txt
   ```
3. **Run Jupyter Notebook**

---

🚀 **Built for Data Science & Marketing Analytics** | 📈 **Powered by Unsupervised Learning**


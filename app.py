# iMPORTING LIBRARIES

from operator import index
import streamlit as st
import plotly.express as px
from pycaret.regression import setup, compare_models, pull, save_model, load_model
from ydata_profiling import ProfileReport
import pandas as pd
from streamlit_pandas_profiling import st_profile_report
import os 


# CHECKING FOR SOURCE 
if os.path.exists("sourcedata.csv"):
    df = pd.read_csv("sourcedata.csv", index_col=None)
    df2 = df


# IMPLEMENTING TITLE AND SIDEBAR

st.title("Reviews Analyser")

with st.sidebar:
    choice = st.radio("Navigation", ["Data Cleaning","Data upload & Profiling", "MODELS","Trial"])


# DATA CLEANING

if choice == "Data Cleaning":

    st.subheader("Data Cleaning")
    st.write("Comming soon...")



# PROFILING OF DATA


if choice == "Data upload & Profiling":

    #Taking Input file From USER (MUST BE CSV)
    st.subheader("Note: Your data must be cleaned and all the reviews should be present into 'Content' column. This application can only be used with application reviews. It wont provide fine results for any other types of results.")
    file = st.file_uploader("Upload File!!")
    if file:
        df = pd.read_csv(file, index_col=None)
        st.dataframe(df)
        df.to_csv("sourcedata.csv", index=None)

        #Generating Profile Report
        report = df.profile_report()
        st_profile_report(report)


if choice == "Trial":


    #SUMMARY 


    from summa import summarizer

    st.subheader("Summary: ")

    text = ' '.join(df['Content'].tolist())
    summaryresult = summarizer.summarize(text)
    st.write(summaryresult)






    #ANOMALLY REVIEWS

    st.subheader("Anamoly reviews: ")

    import pandas as pd
    from sklearn.neighbors import LocalOutlierFactor



    # Vectorize the text using the TF-IDF algorithm
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['Content'])

    # Detect anomalies using the LOF algorithm
    lof = LocalOutlierFactor(n_neighbors=20)
    anomalies = lof.fit_predict(X)

    # Print the indices of the anomalous reviews
    anomalous_indices = [i for i, x in enumerate(anomalies) if x == -1]

    st.write(anomalous_indices)






# MODELS FOR REVIEWS


if choice =="MODELS":

    #IMPORTING LIBRARIES
    from textblob import TextBlob


    #Printing DATA

    st.subheader("Peek into Data: ")
    st.write(df)



    #SENTIMENTAL ANALYSIS

    st.subheader("Sentiments: ")

    # Create a new column for sentiment analysis results
    df['Sentiment'] = ''

    # Iterate over each review and perform sentiment analysis
    for index, row in df.iterrows():
        review = row['Content']
        blob = TextBlob(review)
        sentiment = blob.sentiment.polarity
        # Classify sentiment as positive, negative, or neutral
        if sentiment > 0:
            df.at[index, 'Sentiment'] = 'Positive'
        elif sentiment < 0:
            df.at[index, 'Sentiment'] = 'Negative'
        else:
            df.at[index, 'Sentiment'] = 'Neutral'


    df['Sentiment'].value_counts() 
    sentimentcount = df['Sentiment'].value_counts()

    st.write(df)
    st.write("Sentiment Counts: ")
    st.write(sentimentcount)


    # KEYWORD FILTERING 

    import re

    st.subheader("KeyWords Filtering: ")

        # Define the rules
    rules = {
        "user_interface": ["user interface", "UI"],
        "functionality": ["functionality", "features"],
        "performance": ["performance", "speed"],
        "customer_support": ["customer support", "support team"],
        "pricing": ["price", "cost", "subscription", "free"],
        "ease_of_use": ["intuitive", "user-friendly", "easy to navigate"],
        "bugs_errors": ["crash", "freeze", "bug", "glitch"],
        "updates": ["update", "upgrade", "new features", "bug fixes"],
        "integration": ["compatibility", "integration", "sync", "connect"],
        "security_privacy": ["secure", "private", "data protection", "encryption"],
    }

    # Define the function
    def rule_based_sentiment(review):
        aspects = {}
        for aspect, keywords in rules.items():
            aspect_sentiment = None  # Initialize aspect sentiment as None
            for keyword in keywords:
                if re.search(keyword, review, re.IGNORECASE):
                    aspect_sentiment = "positive"
                    break  # Stop searching for keywords for this aspect if a positive match is found
                elif re.search(r"(not|n't)\s*\b{}\b".format(keyword), review, re.IGNORECASE):
                    aspect_sentiment = "negative"
                    break  # Stop searching for keywords for this aspect if a negative match is found
            if aspect_sentiment is not None:
                aspects[aspect] = aspect_sentiment
        return aspects





    # Apply the function to each review and store the results in a new column
    df["aspects_sentiments"] = df["Content"].apply(rule_based_sentiment)
    aspectsentiments = df["aspects_sentiments"].value_counts() 


    st.write(df)
    st.write("Aspect Sentiments and count: ")
    st.write(aspectsentiments)



    #FREQUENCY OF WORDS

    st.subheader("Most Frequent words: (Descending Order)")

    from collections import Counter

    def create_freq_count(df, column):

        # Concatenate all the reviews into a single string
        reviews = " ".join(df[column].astype(str).tolist())

        # Tokenize the string into words and count their frequency
        words = reviews.split()
        freq_count = dict(Counter(words))

        # Convert the frequency count to a pandas DataFrame
        df_freq_count = pd.DataFrame(list(freq_count.items()), columns=['word', 'frequency'])

        return df_freq_count

    # Create a frequency count of the "review_text" column
    freq_count = create_freq_count(df, 'Content')

    # Print the top 10 most frequent words in the reviews
    print(freq_count.nlargest(15, 'frequency'))

    frequencycount = freq_count.nlargest(15, 'frequency')

    st.write(frequencycount)






    # TOP COMMENTS



    # Create a checkbox
    st.subheader("Top Comments based on Thumbs up count: ")
    checkbox_value = st.checkbox('Check only if your csv contains "Thumns up" column.')

    # Use the checkbox value in your code
    if checkbox_value:
        topcomments = {"Thumbs up counts": df.tail(5)['Thumbs up'].tolist(), "Reviews": df.tail(5)['Content'].tolist()}
        df_topcomments = pd.DataFrame(topcomments)

        st.write(df_topcomments)



    # COMPLAINTS IDENTIFICATION (RULES)


    st.subheader("Top complaints: ")


    # Filter the reviews to include only negative and neutral sentiment reviews
    negative_reviews = df[df['Sentiment'].isin(['Negative', 'Neutral'])]

    # Create an empty list to hold the complaints
    complaints = []

    # Loop through the negative and neutral reviews and look for common complaints
    for review in negative_reviews['Content']:
        # Convert the review text to lowercase and split it into individual words
        words = review.lower().split()
        # Look for specific words or phrases that indicate a complaint
        if 'crash' in words:
            complaints.append('Crashes')
        if 'freeze' in words:
            complaints.append('Freezes')
        if 'slow' in words:
            complaints.append('Slow performance')
        if 'bug' in words:
            complaints.append('Bugs')
        if 'compatibility' in words:
            complaints.append('Compatibility issues')
        if 'login' in words or 'authentication' in words:
            complaints.append('Login or authentication problems')
        if 'ui' in words or 'ux' in words or 'design' in words:
            complaints.append('UI/UX design issues')
        if 'feature' in words or 'functionality' in words or 'option' in words or 'tool' in words:
            complaints.append('Missing features')
        if 'broken' in words or 'error' in words:
            complaints.append('Broken links or error messages')
        if 'navigation' in words or 'menu' in words:
            complaints.append('Navigation problems')
        if 'label' in words or 'tag' in words or 'name' in words:
            complaints.append('Inconsistent or confusing labeling')
        if 'confusing' in words or 'unclear' in words or 'ambiguous' in words:
            complaints.append('Confusing or unhelpful error messages')
        if 'outdated' in words or 'obsolete' in words:
            complaints.append('Outdated information or content')
        if 'instruction' in words or 'guide' in words or 'manual' in words:
            complaints.append('Poorly written or unclear instructions')
        if 'security' in words or 'hack' in words or 'virus' in words:
            complaints.append('Inadequate security measures or Malware/viruses')
        if 'connect' in words or 'connectivity' in words or 'connection' in words:
            complaints.append('Connection issues')
        if 'data' in words or 'backup' in words:
            complaints.append('Data loss or corruption')
        if 'hardware' in words or 'device' in words:
            complaints.append('Hardware compatibility issues')
        if 'hardware' in words or 'device' in words or 'compatibility' in words:
                complaints.append('Hardware compatibility issues')
        if 'battery' in words or 'drain' in words:
                complaints.append('Battery drain')
        if 'overheat' in words or 'overheating' in words:
                complaints.append('Overheating')
        if 'customization' in words:
            if 'limited' in words or 'non-existent' in words:
                complaints.append('Limited or non-existent customization options')
        if 'search' in words:
            if 'poor' in words or 'non-functional' in words or 'design' in words:
                complaints.append('Poorly designed or non-functional search functionality')
        # Add additional if statements for other common complaints
        
    # Count the frequency of each complaint
    complaint_counts = Counter(complaints)
    df_complaints = pd.DataFrame(complaint_counts, index=[1])
    st.write(df_complaints)



    #CLUSTERING ANALYSIS

    st.subheader("Clustering results:")
    st.write("Each review is assigned to cluster")
    st.write("Number of Clusters: 5")



    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
     



    # Convert the preprocessed text to numerical vectors using TF-IDF vectorizer
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['Content'])

    # Cluster the numerical vectors using KMeans clustering algorithm
    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans.fit(X)

    # Add the cluster labels to the DataFrame
    df['Cluster'] = kmeans.labels_

    # Counting cluster entries

    cluster_entries = df['Cluster'].value_counts()


    st.write(df)
    st.write("Cluster Entries count: (Ascending order)")
    st.write(cluster_entries)



    # TOPIC MODELLING USING LDA

    st.subheader("Topic Modelling using LDA: ")

    from gensim.utils import simple_preprocess
    from gensim.corpora import Dictionary
    from gensim.models.ldamodel import LdaModel

    # Load the reviews from a pandas DataFrame
    reviews = df2['Content'].values.tolist()

    # Tokenize and preprocess the reviews
    def preprocess(review):
        return simple_preprocess(review, deacc=True)

    processed_reviews = [preprocess(review) for review in reviews]

    # Create a dictionary of the words in the reviews
    dictionary = Dictionary(processed_reviews)

    # Create a bag of words representation for each review
    corpus = [dictionary.doc2bow(review) for review in processed_reviews]

    # Train the LDA model
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=10, random_state=100, 
                        update_every=1, chunksize=100, passes=10, alpha='auto', per_word_topics=True)

    # Iterate over each topic and add it to the DataFrame
    topics_df = pd.DataFrame()
    for idx, topic in lda_model.show_topics(num_topics=10, num_words=10, formatted=False):
        # Get the top 10 words for the topic
        words = [w[0] for w in topic]
        # Add the topic and its words to the DataFrame
        topics_df = topics_df.append({'Topic': idx, 'Words': words}, ignore_index=True)

    st.write(topics_df)





    # NETWORK ANALYSIS

    st.subheader("Network analysis: ")



    import networkx as nx

    # Load the reviews from a pandas DataFrame
    total_reviews = df['Content'].values.tolist()

    # Create a network graph based on co-occurrences of words in the reviews
    G = nx.Graph()
    for review in total_reviews:
        words = review.split()
        for i in range(len(words)):
            if words[i] not in G.nodes():
                G.add_node(words[i])
            for j in range(i+1, len(words)):
                if words[j] not in G.nodes():
                    G.add_node(words[j])
                if G.has_edge(words[i], words[j]):
                    G[words[i]][words[j]]['weight'] += 1
                else:
                    G.add_edge(words[i], words[j], weight=1)

    # Print the nodes and edges with the highest weights
    top_nodes = sorted(G.nodes(), key=lambda x: G.degree(x), reverse=True)[:10]
    top_edges = sorted(G.edges(), key=lambda x: G[x[0]][x[1]]['weight'], reverse=True)[:10]

    

    # network graph
    import matplotlib.pyplot as plt

    # Draw the network graph
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, nodelist=top_nodes, node_size=100, node_color='r')
    nx.draw_networkx_edges(G, pos, edgelist=top_edges, width=2, edge_color='b')
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')

    # Show the plot
    plt.axis('off')
    st.pyplot(plt.gcf())



    # ASSOCIATION RULE MINING

    st.subheader("Accociation Rules: ")

   
    from mlxtend.preprocessing import TransactionEncoder
    from mlxtend.frequent_patterns import apriori, association_rules

    # Load the reviews from a pandas DataFrame
    df2 = df
    reviews = df['Content'].apply(lambda x: x.split()).values.tolist()

    # Encode the reviews as a one-hot matrix
    te = TransactionEncoder()
    te_ary = te.fit(reviews).transform(reviews)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    # Find frequent itemsets using the Apriori algorithm
    frequent_itemsets = apriori(df, min_support=0.05, use_colnames=True)

    # Generate association rules based on the frequent itemsets
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

    # Print the top 10 rules sorted by lift
    top_associations = rules.sort_values(by=['lift'], ascending=False).head(10)
    
    st.write(top_associations)




    # TOPIC MODELLING USING LSA

    st.subheader("Topic Modelling using LSA: ")

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD

    # Load the reviews from a pandas DataFrame
    LSA_reviews = df2['Content'].values.tolist()

    # Vectorize the reviews using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(LSA_reviews)

    # Perform LSA on the vectorized reviews
    lsa = TruncatedSVD(n_components=10, random_state=42)
    X_lsa = lsa.fit_transform(X)

    def get_top_terms(num_topics=10, num_terms=10):
        terms = vectorizer.get_feature_names_out()
        top_terms = []
        for i, comp in enumerate(lsa.components_):
            terms_in_comp = zip(terms, comp)
            sorted_terms = sorted(terms_in_comp, key=lambda x: x[1], reverse=True)[:num_terms]
            topic_terms = [term[0] for term in sorted_terms]
            top_terms.append(topic_terms)
        df = pd.DataFrame(top_terms, index=[f"Topic {i}" for i in range(num_topics)])
        return df

    df = get_top_terms(num_topics=10, num_terms=10)
    st.dataframe(df)




    # WORD CLOUD

    st.subheader("Word cloud: ")

    from wordcloud import WordCloud
    import matplotlib.pyplot as plt

    # Combine all reviews into a single string
    reviews_combined = " ".join(df2["Content"])

    # Generate a word cloud image
    wordcloud = WordCloud(width=800, height=800, background_color='white', max_words=100, collocations=False).generate(reviews_combined)

    # Display the word cloud image
    plt.figure(figsize=(8,8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot()




    

    


        










        





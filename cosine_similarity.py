from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_cosine_similarity(text1, text2):
    texts = [text1, text2]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)

    cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    
    return cosine_sim[0][0]

text1 = "Her sabah işe gitmeden önce kahvaltı yaparım ve gazete okurum."
text2 = "Her gün işe başlamadan önce kahvaltımı yapar ve gazetedeki haberleri okurum."

similarity_same_meaning = calculate_cosine_similarity(text1, text2)
print(f"Cosine Similarity (same meaning): {similarity_same_meaning}")

text3 = "Her sabah işe gitmeden önce kahvaltı yaparım ve gazete okurum."
text4 = "Akşamları işten sonra spor salonuna giderim ve ağırlık kaldırırım."

similarity_different_meaning = calculate_cosine_similarity(text3, text4)
print(f"Cosine Similarity (different meaning): {similarity_different_meaning}")

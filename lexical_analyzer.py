##parser✅
##eliza✅
##n-gram✅
##tf-idf ✅
#word2vec✅
#doctor✅

import csv
import re
import random
import math
from collections import Counter

class Lexer:
    def __init__(self): 
        self.analyses = { 
            1: self.show_phrases,
            2: self.term_count,
            3: self.eliza_chat,
            4: self.n_gram_analysis,
            5: self.tfidf_analysis,
            6: self.word2vec_training,#convert to vector
            7: self.doctor_chat,
        } #map analysis types to their corresponding methods

##parser        
    def show_phrases(self, phrases):
        print("Phrases:")
        for phrase_type, phrase in phrases:
            phrase_str = ' '.join([word for word, tag in phrase])
            print(f"Document: {phrase_type}, Phrase: {phrase_str}")
            
            
    def term_count(self, phrases):
        print("Term Count:")
        np_count = 0
        vp_count = 0
        for _, phrase_type_list in phrases:
            phrase_type = phrase_type_list[0][0]  
            if phrase_type == "NP":
                np_count += 1
            elif phrase_type == "VP":
                vp_count += 1
        print(f"Number of Noun Phrases (NP): {np_count}")
        print(f"Number of Verb Phrases (VP): {vp_count}")
            

#eliza
    def eliza_chat(self, phrases):
        print("ELIZA Analysis:")
        for full_phrase, phrase_details in phrases: #analysed
            print(f"\nAnalyzing Phrase: {full_phrase}")
            response = self.eliza_response(full_phrase) #response
            print(f"ELIZA's Response: {response}")

    def eliza_response(self, statement):

#ELIZA's rules
        rules = [
            (r'I need (.*)', ["Why do you need {}?", "Would it really help you to get {}?", "Are you sure you need {}?"]),
            (r'I\'m (.*)', ["Why do you think you're {}?", "Do you enjoy being {}?", "Why do you tell me you're {}?", "How long have you been {}?"]),
            (r'Do you (.*)\?', ["Why does it matter whether I {}?", "Would you prefer it if I {}?", "Maybe you believe I {}.", "I may {} -- what do you think?"]),
            (r'(.*)', ["Please tell me more.", "Let's change focus a bit... Tell me about your family.", "Can you elaborate on that?", "Why do you say that {}?", "I see.", "Very interesting.", "I see.  And what does that tell you?", "How does that make you feel?", "How do you feel when you say that?"])
        ]

        for pattern, responses in rules:
            match = re.match(pattern, statement.rstrip(".!"))
            if match:
                response = random.choice(responses)
                return response.format(*[self.reflect(g) for g in match.groups()])

        return "Tell me more about that."

    def reflect(self, fragment):

        tokens = fragment.lower().split()
        reflections = {
            "i": "you", "you": "I", "me": "you", "your": "my", "yours": "mine", "my": "your", "am": "are", "are": "am"
        }
        reflected_tokens = [reflections.get(word, word) for word in tokens]
        return ' '.join(reflected_tokens)


#n-gram  
    def n_gram_analysis(self, phrases, n=2):
        print(f"\n{n}-Gram Analysis:\n")
        n_grams = []
        for full_phrase, _ in phrases:
            tokens = full_phrase.split()
            for i in range(len(tokens) - n + 1):
                n_gram = ' '.join(tokens[i:i + n])
                n_grams.append(n_gram)
        
        n_gram_counts = Counter(n_grams)
        for n_gram, count in n_gram_counts.items():
            print(f"{n_gram}: {count}")


#tfidf
    def tfidf_analysis(self, phrases):
        print("\nTF-IDF Analysis:")
        documents = [phrase for phrase, _ in phrases]
        query = input("Enter your query: ")
        scores = {doc: self.score_query(query, doc, documents) for doc in documents}
        ranked_documents = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        for doc, score in ranked_documents:
            print(f"Document: {doc}\nScore: {score:.4f}\n")

    def compute_tf(self, word, doc):
        frequency = doc.count(word)
        return frequency / len(doc.split())

    def compute_idf(self, word, docs):
        return math.log(len(docs) / sum([1 for doc in docs if word in doc]))

    def compute_tf_idf(self, doc, docs):
        tf_idf = {}
        for word in set(doc.split()):
            tf = self.compute_tf(word, doc)
            idf = self.compute_idf(word, docs)
            tf_idf[word] = tf * idf
        return tf_idf

    def score_query(self, query, doc, docs):
        score = 0.0
        for word in query.split():
            if word in doc:
                score += self.compute_tf_idf(doc, docs).get(word, 0)
        return score


#word2vec
    def word2vec_training(self, phrases):
        print("\nWord2Vec Analysis:")
        sentences = [phrase.lower() for phrase, _ in phrases]
        vocab = self.preprocess(sentences)
        vocab_size = len(vocab)
        word_index = {w: i for i, w in enumerate(vocab)}
        index_word = {i: w for w, i in word_index.items()}
        
        embedding_size = 2
        input_layer = [[random.uniform(-1, 1) for _ in range(embedding_size)] for _ in range(vocab_size)]
        output_layer = [[random.uniform(-1, 1) for _ in range(vocab_size)] for _ in range(embedding_size)]

        learning_rate = 0.01
        epochs = 50
        window_size = 2

        for epoch in range(epochs):
            for sentence in sentences:
                words = sentence.split()
                for i, word in enumerate(words):

                    start = max(0, i - window_size)
                    end = min(len(words), i + window_size + 1)
                    context_words = words[start:i] + words[i+1:end]

        word_embeddings = {w: input_layer[word_index[w]] for w in vocab}
        print("Word Embeddings:")
        for word, embedding in word_embeddings.items():
            print(f"{word}: {embedding}")

    def preprocess(self, sentences):
        words = set()
        for sentence in sentences:
            for word in sentence.split():
                words.add(word)
        return list(words)


#doctor
    def doctor_chat(self, phrases):
        print("Doctor Analysis:")
        print("DOCTOR: Good day. I'm Doctor. What seems to be the problem?")

        processed_sentences = set()  
        for full_phrase, _ in phrases:
            if full_phrase not in processed_sentences:
                print("Patient:", full_phrase)
                response = self.doctor_response(full_phrase.lower())
                print("DOCTOR:", response)
                processed_sentences.add(full_phrase)
    def doctor_response(self, user_input):
        patterns = {
            r'(.*)\bheadache\b(.*)': ["It seems like you have a headache. Have you taken any painkillers?", "Headaches can be uncomfortable. Have you tried resting in a quiet, dark room?"],
            r'(.*)\bstomachache\b(.*)': ["Stomach aches can have various causes. Have you eaten anything unusual recently?", "A stomachache could be due to digestive issues. Have you made any changes to your diet?"],
            r'(.*)\bfever\b(.*)': ["Fever often indicates infection. Have you had any other symptoms such as cough or sore throat?", "Fever can be the body's response to infection. Have you taken any medication?"],
            r'(.*)\bcough\b(.*)': ["A persistent cough could indicate respiratory issues. Have you experienced shortness of breath?", "Coughing can be bothersome. Have you tried remedies like honey or cough syrup?"],
            r'(.*)\bsore throat\b(.*)': ["A sore throat may be a sign of a viral infection. Have you been around sick people?", "Sore throats can be unpleasant. Have you tried gargling with warm salt water?"],
            r'(.*)': ["I understand. Please provide more details about your symptoms.", "Let's discuss further. Can you describe how you're feeling?", "Your health is important. Please elaborate on your symptoms."]
        }
    
        
        for pattern, responses in patterns.items():
            if re.search(pattern, user_input, re.IGNORECASE):
                return random.choice(responses)
        return "It seems I'm not equipped to diagnose this. Consider consulting a medical professional."


#menu
    def process_output(self, parser_output):
        phrases = parser_output
        while True:
            print("\nAvailable analyses:")
            print("1. Show Phrases From Parser")
            print("2. Term Count Analysis")
            print("3. ELIZA Analysis")
            print("4. N-Gram Analysis")
            print("5. TF-IDF Analysis")
            print("6. Word2Vec Analysis")
            print("7. Doctor Analysis")
            print("8. Exit")
            choice = input("Enter your choice: ")

            if choice == '1':
                self.show_phrases(phrases)
            elif choice == '2':
                self.term_count(phrases)
            elif choice == '3':
                self.eliza_chat(phrases)
            elif choice == '4':
                self.n_gram_analysis(phrases)
            elif choice == '5':
                self.tfidf_analysis(phrases)
            elif choice == '6':
                self.word2vec_training(phrases)
            elif choice == '7':
                self.doctor_chat(phrases)
            elif choice == '8':
                print("Exiting!")
                break
            else:
                print("Invalid choice. Please enter a number between 1 and 8.")


            
lexer = Lexer()

input_data = []

#csv file from parser
with open('parser.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        phrase_type = row[0]
        phrase = row[1].split()  
        phrase_list = [(word, None) for word in phrase]  
        input_data.append((phrase_type, phrase_list))

print(input_data)

lexer.process_output(input_data)

import PyPDF2
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
import matplotlib.pyplot as plt


def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def preprocess_text(text):
    sentences = sent_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    cleaned_sentences = []

    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        cleaned_words = [lemmatizer.lemmatize(word.lower()) for word in words if word.isalpha() and word.lower() not in stop_words]
        cleaned_sentences.append(cleaned_words)

    return cleaned_sentences


def extract_questions(sentences):
    questions = []
    for sentence in sentences:
        if len(sentence) >= 2 and sentence[0].isdigit() and sentence[1] == '.':
            questions.append(' '.join(sentence[1:]))
    return questions


def compare_questions(questions1, questions2):
    flattened_questions1 = [question.lower() for sublist in questions1 for question in sublist]
    flattened_questions2 = [question.lower() for sublist in questions2 for question in sublist]
    question_counts1 = Counter(flattened_questions1)
    question_counts2 = Counter(flattened_questions2)
    common_questions = question_counts1 & question_counts2
    return common_questions


def calculate_insights(common_questions, flattened_questions1, flattened_questions2, num_top_questions=10):
    if not common_questions:
        print("No common questions found.")
        return

    top_questions = common_questions.most_common(num_top_questions)
    total_questions = sum(common_questions.values())

    if len(flattened_questions1) + len(flattened_questions2) > 0:
        percentage_common = (total_questions / (len(flattened_questions1) + len(flattened_questions2))) * 100
        print(f"Total common questions: {total_questions}")
        print(f"Percentage of common questions: {percentage_common:.2f}%")
    else:
        print("No questions found.")

    print("Top repeated questions:")
    for question, count in top_questions:
        print(f"Question: {question}\tCount: {count}")

    questions, counts = zip(*common_questions.items())
    plt.figure(figsize=(10, 6))
    plt.barh(questions, counts)
    plt.xlabel('Count')
    plt.ylabel('Question')
    plt.title('Distribution of Question Repetitions')
    plt.tight_layout()
    plt.show()


def main():
    # Example usage: Extract text from two PDFs
    pdf1_text = extract_text_from_pdf('paper1.pdf')
    pdf2_text = extract_text_from_pdf('paper3.pdf')

    # Example usage: Preprocess the text from two PDFs
    pdf1_sentences = preprocess_text(pdf1_text)
    pdf2_sentences = preprocess_text(pdf2_text)

    # Example usage: Extract questions from preprocessed sentences
    pdf1_questions = extract_questions(pdf1_sentences)
    pdf2_questions = extract_questions(pdf2_sentences)

    # Example usage: Compare and count the repetitions
    common_questions = compare_questions(pdf1_questions, pdf2_questions)

    # Example usage: Calculate insights and visualize the results
    flattened_questions1 = [question.lower() for sublist in pdf1_questions for question in sublist]
    flattened_questions2 = [question.lower() for sublist in pdf2_questions for question in sublist]
    calculate_insights(common_questions, flattened_questions1, flattened_questions2)


if __name__ == '__main__':
    main()

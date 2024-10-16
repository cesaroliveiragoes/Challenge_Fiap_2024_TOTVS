import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
from nltk.util import ngrams
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Carregar o dataset
df = pd.read_csv("transcricao_pbi.csv", encoding='latin-1')

# Download NLTK datasets
nltk.download('punkt') # Faz o download do pacote de tokenização de palavras
nltk.download('stopwords') # Faz o download do pacote de stopwords (palavras irrelevantes)

# Lista de stopwords em português
stop_words = set(stopwords.words('portuguese'))  # Define a lista de stopwords em português

# Lista adicional de preposições, conjunções, saudações e outras palavras desnecessárias para o modelo
additional_stopwords = {
    'a', 'ante', 'até', 'após', 'com', 'contra', 'de', 'desde', 'em', 'entre', 'para',
    'per', 'perante', 'por', 'sem', 'sob', 'sobre', 'trás', 'e', 'mas', 'ou', 'nem',
    'porque', 'pois', 'tchau', 'oi', 'olá', 'pra', 'tudo', 'bem', 'né', 'bom', 'tá', 'falar', 'minutinhos',
    'ah', 'ai','dia','caso', 'amigo', 'colega', 'hoje', 'pouquinho', 'aí', 'senhor', 'totos', 'nps', 'brasil',
    'bloco', 'então', 'gente','cnpj', 'sarah', 'bali', 'algum', 'nps','dp', 'peço', 'si', 'utiliza', 'nota', 'precisam',
    'agora', 'empresa', 'numero', 'número', 'gostaria', 'mais', 'meu', 'nosso',
    'nossa', 'você', 'vocês', 'eles', 'elas', 'ele', 'ela', 'dele', 'dela',
    'nos', 'nós', 'me', 'te', 'lhe', 'pro', 'sr', 'senhora', 'sra', 'seu',
    'dona', 'nome', 'gostaria', 'uns', 'umas', 'não', 'sim', 'está', 'estão',
    'foi', 'será', 'serão', 'ter', 'tive', 'tiver', 'tinha', 'tem', 'tenho', 'sendo',
    'estive', 'está', 'ficar', 'ficou', 'fiquei', 'podemos', 'poder', 'pode', 'fazer',
    'fazendo', 'feito', 'já', 'agora', 'tempo', 'tudo', 'todos', 'só', 'muito',
    'pouco', 'nada', 'mesmo', 'assim', 'também', 'ainda', 'quero', 'vamos', 'vai',
    'vem', 'veio', 'ir', 'indo', 'indo', 'ver', 'vendo', 'ouvir', 'ouvindo', 'preciso',
    'precisamos', 'precisa', 'faz', 'fazemos', 'quero', 'disse', 'dizer', 'falou',
    'teremos', 'terão', 'tiveram', 'poderemos', 'esperar', 'esperando', 'esperou',
    'ligar', 'ligando', 'ligou', 'mandar', 'mandou', 'enviar', 'enviou', 'estamos',
    'ajudar', 'ajudando', 'ajudou', 'resolver', 'resolvendo', 'resolvido', 'telefone', 'email', 'usuario', 'usuaria',
    'usuário', 'alguma', 'algum', 'alguns', 'totalmente', 'parcialmente','total','parcial', 'usuária', 'chamo',
    'minutos','boletos','boleto','iniciando', 'contato', 'falando','dois', 'comigo','alguém', 'entrega'}

# Lista de nomes comuns em português (exemplo)
common_names = {
    'ana', 'maria', 'joão', 'jose', 'josé', 'antonio', 'francisco', 'carlos', 'paulo',
    'pedro', 'lucas', 'luiz', 'marcos', 'gabriel', 'rafael', 'daniel', 'vinicius',
    'eduardo', 'roberto', 'fernando', 'juliana', 'bruna', 'fernanda', 'camila', 'amanda',
    'jessica', 'leticia', 'beatriz', 'julio', 'rodrigo', 'renato', 'ricardo', 'tiago',
    'thiago', 'felipe', 'luís', 'sara', 'isabela', 'larissa', 'laura', 'giovana',
    'mariana', 'alice', 'hugo', 'luciana', 'marcio', 'sergio', 'andré', 'carla', 'carol',
    'sandra', 'vanessa', 'gabriela', 'priscila', 'patricia', 'aline', 'daniela',
    'andreia', 'roberta', 'bianca', 'eduarda'
}

# Atualizar o conjunto de stopwords
stop_words.update(additional_stopwords) # Adiciona as stopwords personalizadas à lista principal
stop_words.update(common_names) # Adiciona os nomes próprios à lista de stopwords

# Função para tokenizar e remover stopwords e letras únicas
def preprocess_text(text):
    tokens = word_tokenize(text.lower()) # Converte o texto em tokens e coloca tudo em letras minúsculas
    filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words and len(word) > 1] # Remove stopwords, números e palavras de uma letra
    return filtered_tokens

# Preprocessando o texto
df['Filtered_Tokens'] = df['texto'].apply(preprocess_text) # Aplica o pré-processamento de texto à coluna "texto"

# Função para gerar e contar n-grams
def generate_ngrams(tokens_list, n):
    ngram_list = list(ngrams(tokens_list, n)) # Gera uma lista de n-grams (bigrams, trigrams, etc.)
    ngram_counts = Counter(ngram_list) # Conta a frequência de cada n-gram
    return ngram_counts

# Gerando bigrams e trigrams
df['Bigrams'] = df['Filtered_Tokens'].apply(lambda x: generate_ngrams(x, 2)) # Gera bigrams a partir dos tokens filtrados
df['Trigrams'] = df['Filtered_Tokens'].apply(lambda x: generate_ngrams(x, 3)) # Gera trigrams a partir dos tokens filtrados

# Inicializando contadores para bigrams e trigrams
bigram_counts = Counter() # Cria um contador vazio para bigrams
trigram_counts = Counter() # Cria um contador vazio para trigrams

# Somando os contadores
for bigram_counter in df['Bigrams']:
    bigram_counts.update(bigram_counter) # Atualiza o contador de bigrams com os valores da coluna
for trigram_counter in df['Trigrams']:
    trigram_counts.update(trigram_counter) # Atualiza o contador de trigrams com os valores da coluna

# Remover n-grams que contêm stopwords ou palavras de uma letra
def filter_ngrams(ngram_counts):
    filtered_ngram_counts = Counter() # Cria um novo contador para os n-grams filtrados
    for ngram, freq in ngram_counts.items():
        if not any(word in stop_words or len(word) == 1 for word in ngram):
            filtered_ngram_counts[ngram] = freq # Mantém apenas os n-grams que não contêm stopwords ou palavras de uma letra
    return filtered_ngram_counts

# Aplicar filtro nos bigrams e trigrams
bigram_counts = filter_ngrams(bigram_counts) # Aplica o filtro nos bigrams
trigram_counts = filter_ngrams(trigram_counts) # Aplica o filtro nos trigrams

# Função para gerar a word cloud de n-grams
def generate_wordcloud(ngram_counts, title):
    # Criando a string combinada de n-grams para a word cloud
    word_freq = {' '.join(k): v for k, v in ngram_counts.items()}  # Converte os n-grams em string e cria um dicionário de frequências
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq) # Gera a word cloud

    # Plotando a word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title, fontsize=18)
    plt.axis('off')
    plt.show()

# Gerando e exibindo as word clouds
generate_wordcloud(bigram_counts, 'Word Cloud de Bigramas')
generate_wordcloud(trigram_counts, 'Word Cloud de Trigramas')

# Convertendo os bigrams e trigrams para DataFrame e salvando como CSV
bigrams_df = pd.DataFrame(bigram_counts.most_common(20), columns=['Bigram', 'Frequency'])
trigrams_df = pd.DataFrame(trigram_counts.most_common(20), columns=['Trigram', 'Frequency'])

# Salvando os DataFrames como CSV
bigrams_df.to_csv('bigrams.csv', index=False)
trigrams_df.to_csv('trigrams.csv', index=False)

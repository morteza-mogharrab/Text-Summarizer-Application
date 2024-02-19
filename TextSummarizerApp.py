import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

class TextSummarizerApp:
    def __init__(self, master):
        self.master = master  # Initialize the TextSummarizerApp class
        master.title('Text Summarizer')  # Set the title of the GUI window
        master.geometry("600x550")  # Set the initial size of the window
        master.resizable(False, False)  # Disable resizing of the window
        master.configure(background='#FFFFFF')  # Set background color of the window

        # Configure styles for widgets
        self.style = ttk.Style()  # Create a ttk style object
        self.style.configure('TFrame', background='#FFFFFF')  # Configure style for frames
        self.style.configure('TButton', background='#FFFFFF')  # Configure style for buttons
        self.style.configure('TLabel', background='#FFFFFF', font=('Arial', 10))  # Configure style for labels
        self.style.configure('Header.TLabel', font=('Arial', 25, 'bold'))  # Configure style for header label
        self.style.configure('text1.TLabel', font=('Arial', 10, 'bold'))  # Configure style for text label
        self.style.configure('Main.TLabel', font=('Arial', 9), background='#e8e8e8')  # Configure style for main labels
        self.style.configure('Alarm.TButton', foreground='black', font=('Arial', 13, 'bold'), width=20)  # Configure style for alarm button
        self.style.map('Alarm.TButton', foreground=[('pressed', 'red')])  # Map style for alarm button
        self.style.theme_use('vista')  # Use the vista theme

        self.frame_header = ttk.Frame(master)  # Create a frame for header
        self.frame_header.pack()  # Pack the header frame

        self.logo = tk.PhotoImage(file='Logo.png')  # Load the logo image
        ttk.Label(self.frame_header, image=self.logo).grid(row=0, column=0, rowspan=1, columnspan=1)  # Add logo to the header frame
        ttk.Label(self.frame_header, text='Text Summarizer', style='Header.TLabel').grid(row=1, column=0)  # Add header label to the header frame
        ttk.Label(self.frame_header, wraplength=580, style='Main.TLabel',
                  text=("Instructions: \n1 - Specify the desired number of sentences for your summary in the provided text box. \n2 - Select a TXT or CSV file for summarization and allow approximately 1 minute for processing. \n3 - After processing is finished, your final TXT file will be created in the same directory. \n4 - Use the PRINT button to print your summarized file.")).grid(row=2, column=0)  # Add instruction label to the header frame

        self.frame_content = ttk.Frame(master)  # Create a frame for content
        self.frame_content.pack()  # Pack the content frame

        self.entry_sentences = ttk.Entry(self.frame_content, font=('Arial', 13), width=15)  # Create an entry widget for specifying number of sentences
        self.entry_sentences.grid(row=0, column=0, padx=15, pady=(35, 5), sticky='nsew')  # Grid the entry widget in the content frame

        ttk.Button(self.frame_content, text='Browse your file', style='Alarm.TButton', command=self.process_file, width=15).grid(row=0, column=1, sticky='nsew', padx=15, pady=(35, 5))  # Create a button to browse files and grid it in the content frame

        self.btn_print = ttk.Button(self.frame_content, text="Print", style='Alarm.TButton', command=self.print_summary, width=15)  # Create a button to print the summary and grid it in the content frame
        self.btn_print.grid(row=0, column=2, sticky='nsew', padx=15, pady=(35, 5))

        self.progress_label = ttk.Label(self.frame_content, text="", font=('Arial', 10))  # Create a label for progress
        self.progress_label.grid(row=1, column=0, columnspan=3, pady=(0, 10))  # Grid the progress label in the content frame

        self.progress_bar = ttk.Progressbar(self.frame_content, orient='horizontal', length=400, mode='determinate', style='Green.Horizontal.TProgressbar')  # Create a progressbar
        self.progress_bar.grid(row=2, column=0, columnspan=3, padx=15, pady=(0, 10))  # Grid the progress bar in the content frame

        self.frame_content.grid_columnconfigure(0, weight=1)  # Configure grid column
        self.frame_content.grid_columnconfigure(1, weight=1)
        self.frame_content.grid_columnconfigure(2, weight=1)

        self.summarized_text_file = None  # Initialize summarized text file variable

    def print_summary(self):
        if self.summarized_text_file:  # Check if summarized text file exists
            os.startfile(self.summarized_text_file, "print")  # Open the summarized text file for printing
        else:
            messagebox.showwarning("Warning", "No file has been created yet.")  # Show a warning message if no file has been created

    def process_file(self):
        sentence_count = self.entry_sentences.get()  # Get the number of sentences specified by the user
        try:
            sentence_count = int(sentence_count)  # Convert the input to an integer
            if sentence_count <= 0:
                raise ValueError  # Raise an error if the input is not a positive integer
        except ValueError:
            messagebox.showwarning("Warning", "Please enter a positive integer for the number of sentences.")  # Show a warning message if input is invalid
            return

        filename = filedialog.askopenfilename()  # Open a file dialog to select a file
        try:
            if filename:
                if filename.endswith('.txt'):
                    with open(filename, 'r') as file:
                        text = file.read()
                    sentences = sent_tokenize(text)  # Tokenize the text into sentences
                elif filename.endswith('.csv'):
                    df = pd.read_csv(filename, header=None)
                    sentences = [sent_tokenize(s) for s in df[0]]
                    sentences = [y for x in sentences for y in x]  
                else:
                    messagebox.showwarning("Warning", "Unsupported file format. Please select a TXT or CSV file.")
                    return

                if sentences:
                    self.progress_label.config(text="Processing file...")  # Set progress label
                    self.master.update_idletasks()  # Update GUI to show progress

                    total_sentences = len(sentences)  # Calculate total number of sentences

                    word_embeddings = {}  # Initialize word embeddings dictionary
                    with open('glove.6B.100d.txt', encoding='utf-8') as f:
                        for line in f:
                            values = line.split()
                            word = values[0]
                            coefs = np.asarray(values[1:], dtype='float32')
                            word_embeddings[word] = coefs  # Load word embeddings from file

                    clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ", regex=False)
                    clean_sentences = [s.lower() for s in clean_sentences]  # Clean and preprocess sentences

                    stop_words = stopwords.words('english')  # Get English stopwords
                    clean_sentences = [self.remove_stopwords(r.split(), stop_words) for r in clean_sentences]  # Remove stopwords from sentences

                    sentence_vectors = [sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()]) / (len(i.split()) + 0.001) for i in clean_sentences]  # Calculate sentence vectors

                    sim_mat = np.zeros([len(sentences), len(sentences)])  # Initialize similarity matrix
                    for i in range(len(sentences)):
                        for j in range(len(sentences)):
                            if i != j:
                                sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1, 100),
                                                                   sentence_vectors[j].reshape(1, 100))[0, 0]  # Calculate cosine similarity between sentences

                    nx_graph = nx.from_numpy_array(sim_mat)  # Create graph from similarity matrix
                    scores = nx.pagerank(nx_graph)  # Calculate pagerank scores

                    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)  # Rank sentences based on scores

                    dir_path = os.path.dirname(os.path.realpath(__file__))  # Get the directory path
                    self.summarized_text_file = os.path.join(dir_path, "Summarized-Text.txt")  # Set path for summarized text file

                    with open(self.summarized_text_file, "w") as f:  # Write summarized text to file
                        for i in range(min(sentence_count, len(ranked_sentences))):
                            f.write(ranked_sentences[i][1] + "\n")  # Write each sentence to file
                            # Update progress bar value
                            self.progress_bar['value'] = (i + 1) / sentence_count * 100
                            self.master.update_idletasks()  # Update GUI to show progress

                    self.progress_label.config(text="Summary created successfully.")  # Set progress label
                    self.master.update_idletasks()  # Update GUI to show progress
                else:
                    messagebox.showwarning("Warning", "The selected file is empty.")  # Show a warning message if the file is empty
            else:
                messagebox.showwarning("Warning", "No file selected.")  # Show a warning message if no file is selected
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")  # Show an error message if an exception occurs

    @staticmethod
    def remove_stopwords(sen, stop_words):
        return " ".join([i for i in sen if i not in stop_words])  # Remove stopwords from sentences

def main():
    root = tk.Tk()  # Create the main window
    text_summarizer_app = TextSummarizerApp(root)  # Initialize the TextSummarizerApp
    root.mainloop()  # Start the main event loop

if __name__ == "__main__":
    main()

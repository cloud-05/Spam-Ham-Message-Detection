#import all the modules
from tkinter import *
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import LancasterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

BG_COLOR="#89CFF0"
FONT_BOLD="Melvetica %d bold"

class SpamHam:
    def __init__(self):
        #initialize tkinter window
        self.window=Tk()
        self.main_window()
        self.lstem = LancasterStemmer()
        self.tfvec=TfidfVectorizer(stop_words='english')
        self.datafile()

    def datafile(self):
        #get all datas from datafile and load the classifier.
        datafile = pickle.load(open("training_data.pkl","rb"))
        self.message_x = datafile["message_x"]
        self.classifier = datafile["classifier"]
    
    def main_window(self):
        #add title to window and configure it
        self.window.title("Spam Detector")
        self.window.resizable(width=False,height=False)
        self.window.configure(width=520,height=400,bg=BG_COLOR)
    
        #head label for the window heading
        head_label=Label(self.window,bg="#d4afb9",fg="#000",text="FastFindFirm",
                         font=FONT_BOLD%(14),pady=10)
        head_label.place(relwidth=1)
        line = Label(self.window,width=200,bg="#000")
        line.place(relwidth=0.5,relx=0.25,rely=0.08,relheight=0.008)

        #mid_label 
        mid_label=Label(self.window,bg=BG_COLOR,fg="#0000FF",text="Spam Or Ham ? Message Detector",
                        font=FONT_BOLD%(18),pady=10)
        mid_label.place(relwidth=1,rely=0.12)

        #answer label where our prediction about user input message will be displayed
        self.answer=Label(self.window,bg=BG_COLOR,fg="#000",text="Please type message below.",
                        font=FONT_BOLD%(16),pady=10,wraplength=525)
        self.answer.place(relwidth=1,rely=0.30)

        #textbox for user to write msg for checking 
        self.msg_entry=Text(self.window,bg="#FFF",fg="#000",font=FONT_BOLD%(14))
        self.msg_entry.place(relwidth=1,relheight=0.4,rely=0.48)
        self.msg_entry.focus()
        #check button to call the prediction function
        check_button=Button(self.window,text="Check",font=FONT_BOLD%(12),width=8,bg="#000",fg="#FFF",
                            command=lambda: self.on_enter(None))        
        check_button.place(relx=0.40,rely=0.90,relheight=0.08,relwidth=0.20)

    def bow(self,message):
        #bag of words
        #transform user's message to fixed vector length 
        mess_t = self.tfvec.fit(self.message_x)
        message_test=mess_t.transform(message).toarray()
        return message_test
    
    def mess(self,messages):
        message_x = []           
        for me_x in messages:
            #filter out other datas except alphabets
            me_x=''.join(filter(lambda mes:(mes.isalpha() or mes==" ") ,me_x))
            #tokenize or split the messages into respective words
            words = word_tokenize(me_x)
            #stem the words to their root words
            message_x+=[' '.join([self.lstem.stem(word) for word in words])]
        return message_x

    def on_enter(self,event):
        #get the user input from textbox
        msg=str(self.msg_entry.get("1.0","end"))
        #preprocess the message
        message=self.mess([msg])
        #predict the the label i.e. ham or spam for users message
        self.answer.config(fg="#ffffff",text="Your message is : "+
                           ("spam" if self.classifier.predict(self.bow(message)).reshape(1,-1) else "ham"))

    #runwindow
    def run(self):
        self.window.mainloop()

# run the file
if __name__=="__main__":
    app = SpamHam()
    app.run()

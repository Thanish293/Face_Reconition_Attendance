from tkinter import *
from tkinter import ttk
from PIL import Image,ImageTk
from student import Student
import os
from train import Train
from face_recognition import Face_recognition
from attendence_table import attendence

class Face_recognotion_system :
    def __init__(self,root):
        self.root=root
        self.root.geometry("1530x790+0+0")
        self.root.title("Face recognition System")
         #logo
        img=Image.open(r"D:\PROJECTS\Attendence system_yt\images\SJEC.jpg")
        img=img.resize((300,100))
        self.photoimg=ImageTk.PhotoImage(img)
        f_lbl=Label(self.root,image=self.photoimg)
        f_lbl.place(x=0,y=0,width=300,height=100)

         #title
        #img1=Image.open(r"D:\PROJECTS\Attendence system_yt\images\title.png")
        ##img1=img1.resize((700,150))
        #self.photoimg1=ImageTk.PhotoImage(img1)
        #f_lbl=Label(self.root,image=self.photoimg1)
        #f_lbl.place(x=400,y=0,width=700,height=150)
        #img2
        img2=Image.open(r"D:\PROJECTS\Attendence system_yt\images\title.jpg")
        img2=img2.resize((450,150))
        self.photoimg2=ImageTk.PhotoImage(img2)
        f_lbl=Label(self.root,image=self.photoimg2)
        f_lbl.place(x=1100,y=0,width=450,height=150)
        
        #bg
        img3=Image.open(r"D:\PROJECTS\Attendence system_yt\images\bg.jpeg")
        img3=img3.resize((1530,710))
        self.photoimg3=ImageTk.PhotoImage(img3)

        bg_img=Label(self.root,image=self.photoimg3)
        bg_img.place(x=0,y=130,width=1530,height=710)
        
        #student button
        img4=Image.open(r"D:\PROJECTS\Attendence system_yt\images\image.png")
        img4=img4.resize((60,60))
        self.photoimg4=ImageTk.PhotoImage(img4)     
        b1=Button(bg_img,image=self.photoimg4,command=self.student_details,cursor="hand2")
        b1.place(x=500,y=150,width=60,height=60)
        b1l=Button(bg_img,text="Student details",command=self.student_details,cursor="hand2",font=("times new roman",15,"bold"),bg="white",fg="black")
        b1l.place(x=560,y=150,width=200,height=60)

        #face button
        img5=Image.open(r"D:\PROJECTS\Attendence system_yt\images\image.png")
        img5=img5.resize((60,60))
        self.photoimg5=ImageTk.PhotoImage(img5)
        b1=Button(bg_img,image=self.photoimg5,cursor="hand2",command=self.face_data)
        b1.place(x=500,y=220,width=60,height=60)
        b1l=Button(bg_img,text="Face Detector",cursor="hand2",command=self.face_data,font=("times new roman",15,"bold"),bg="white",fg="black")
        b1l.place(x=560,y=220,width=200,height=60)

        #attendence
        img6=Image.open(r"D:\PROJECTS\Attendence system_yt\images\image.png")
        img6=img6.resize((60,60))
        self.photoimg6=ImageTk.PhotoImage(img6)
        b1=Button(bg_img,image=self.photoimg6,cursor="hand2",command=self.attendence_data)
        b1.place(x=500,y=290,width=60,height=60)
        b1l=Button(bg_img,text="Attendence",cursor="hand2",command=self.attendence_data,font=("times new roman",15,"bold"),bg="white",fg="black")
        b1l.place(x=560,y=290,width=200,height=60)

        
        #tranining
        img7=Image.open(r"D:\PROJECTS\Attendence system_yt\images\image.png")
        img7=img7.resize((60,60))
        self.photoimg7=ImageTk.PhotoImage(img7)
        b1=Button(bg_img,image=self.photoimg7,cursor="hand2",command=self.train_data)
        b1.place(x=500,y=370,width=60,height=60)
        b1l=Button(bg_img,text="Training",cursor="hand2",command=self.train_data,font=("times new roman",15,"bold"),bg="white",fg="black")
        b1l.place(x=560,y=370,width=200,height=60)

        #Photos
        img8=Image.open(r"D:\PROJECTS\Attendence system_yt\images\image.png")
        img8=img8.resize((60,60))
        self.photoimg8=ImageTk.PhotoImage(img8)
        b1=Button(bg_img,image=self.photoimg8,cursor="hand2",command=self.open_img)
        b1.place(x=500,y=440,width=60,height=60)
        b1l=Button(bg_img,text="Photos",cursor="hand2",command=self.open_img,font=("times new roman",15,"bold"),bg="white",fg="black")
        b1l.place(x=560,y=440,width=200,height=60)
    
    def open_img(self):
        os.startfile("data")

    #============function buttons========
    def student_details(self):
        self.new_window=Toplevel(self.root)
        self.app=Student(self.new_window)

    def train_data(self):
        self.new_window=Toplevel(self.root)
        self.app=Train(self.new_window)

    def face_data(self):
        self.new_window=Toplevel(self.root)
        self.app=Face_recognition(self.new_window)

    def attendence_data(self):
        self.new_window=Toplevel(self.root)
        self.app=attendence(self.new_window)
if __name__ == "__main__":
     root=Tk()
     obj= Face_recognotion_system(root)
     root.mainloop()

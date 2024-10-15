from tkinter import *
from tkinter import ttk
from PIL import Image,ImageTk
from tkinter import messagebox
import mysql.connector
import cv2
import os

class Student :
    def __init__(self,root):
        self.root=root
        self.root.geometry("1530x790+0+0")
        self.root.title("Face recognition System")

        #=========variables============
        self.var_dept=StringVar()
        self.var_USN=StringVar()
        self.var_year=StringVar()
        self.var_name=StringVar()





         #logo
        img=Image.open(r"D:\PROJECTS\Attendence system_yt\images\SJEC.jpg")
        img=img.resize((400,150))
        self.photoimg=ImageTk.PhotoImage(img)
        f_lbl=Label(self.root,image=self.photoimg)
        f_lbl.place(x=0,y=0,width=400,height=150)

         #title
        img1=Image.open(r"D:\PROJECTS\Attendence system_yt\images\title.png")
        img1=img1.resize((700,150))
        self.photoimg1=ImageTk.PhotoImage(img1)
        f_lbl=Label(self.root,image=self.photoimg1)
        f_lbl.place(x=400,y=0,width=700,height=150)
        #img2
        img2=Image.open(r"D:\PROJECTS\Attendence system_yt\images\tittle.png")
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

        title_lb1=Label(bg_img,text="Student Details",font=("times new roman",35,"bold"),bg="Red",fg="white")
        title_lb1.place(x=0,y=0,width=1530,height=45)

        main_frame=Frame(bg_img,bd=2)
        main_frame.place(x=5,y=50,width=1500,height=600)

        #left panel
        Left_frame=LabelFrame(main_frame,bd=2,bg="white",relief=RIDGE,text="Student Details",font=("times new roman",12,"bold"))
        Left_frame.place(x=10,y=10,width=760,height=580)

        Current_frame=LabelFrame(Left_frame,bd=2,bg="white",relief=RIDGE,text="Course Info",font=("times new roman",12,"bold"))
        Current_frame.place(x=5,y=0,width=720,height=200)

        #department
        dept_label=Label(Current_frame,text="Department",font=("times new roman",12,"bold"))
        dept_label.grid(row=0,column=0)
        dept_combo=ttk.Combobox(Current_frame,textvariable=self.var_dept,font=("times new roman",12,"bold"),width=20,state="read only")
        dept_combo["values"]=("Select Deparment","ECE","CSE","EEE","MECH","CIVIL")
        dept_combo.current(0)
        dept_combo.grid(row=0,column=1,padx=2,pady=10,sticky=W)

        
        # Course/year
        course_label=Label(Current_frame, text="Course", font=("times new roman", 13, "bold"))
        course_label.grid(row=0,column=2,padx=10,sticky=W)
        course_combo=ttk.Combobox(Current_frame,textvariable=self.var_year, font=("times new roman", 13, "bold"), state="readonly ",width=20)
        course_combo["values"]=("Select Year", "First", "Second", "Third", "fourth")
        course_combo.current(0)
        course_combo.grid(row=0,column=3, padx=2, pady=10, sticky=W)

        class_student_frame=LabelFrame(Left_frame,bd=2,bg="white",relief=RIDGE,text="Student info",font=("times new roman",12,"bold"))
        class_student_frame.place(x=5,y=200,width=720,height=300)

        #USN
        studentid_label=Label(class_student_frame,text="USN:",font=("times new roman",12,"bold"))
        studentid_label.grid(row=0,column=0)
        studentid_entry=ttk.Entry(class_student_frame,textvariable=self.var_USN,width=20,font=("times new roman",13,"bold"))
        studentid_entry.grid(row=0,column=1,padx=10,sticky=W)

        #name
        studentname_label=Label(class_student_frame,text="Name:",font=("times new roman",12,"bold"))
        studentname_label.grid(row=0,column=2)
        studentname_entry=ttk.Entry(class_student_frame,textvariable=self.var_name,width=20,font=("times new roman",13,"bold"))
        studentname_entry.grid(row=0,column=3,padx=10,sticky=W)

        #radio
        self.var_radio1=StringVar()
        radiobtn1=ttk.Radiobutton(class_student_frame,variable=self.var_radio1,text="Take a photo sample",value="yes")
        radiobtn1.grid(row=7,column=0,pady=10)

       
        radiobtn2=ttk.Radiobutton(class_student_frame,variable=self.var_radio1,text="No photo sample",value="no")
        radiobtn2.grid(row=9,column=0,pady=10)

        #buttons

        btn_frame=Frame(class_student_frame,bd=2,relief=RIDGE)
        btn_frame.place(x=0,y=130,width=715,height=145)

        #save button
        save_btn=Button(btn_frame,text="Save",command=self.add_data,width=20,font=("times new roman",12,"bold"),bg="blue",fg="white")
        save_btn.grid(row=0,column=0)

        #update
        update_btn=Button(btn_frame,text="Update",width=20,command=self.update_data,font=("times new roman",12,"bold"),bg="blue",fg="white")
        update_btn.grid(row=0,column=1)

        #delete
        delete_btn=Button(btn_frame,text="Delete",command=self.delete_data,width=20,font=("times new roman",12,"bold"),bg="blue",fg="white")
        delete_btn.grid(row=0,column=2)

        #reset
        delete_btn=Button(btn_frame,text="Reset",command=self.reset_data,width=20,font=("times new roman",12,"bold"),bg="blue",fg="white")
        delete_btn.grid(row=0,column=3)

        btn_frame2=Frame(class_student_frame,bd=2,relief=RIDGE)
        btn_frame2.place(x=0,y=175,width=715,height=100)
        #photo button
        take_photo_btn=Button(btn_frame2,text="Take a photo sample",command=self.generate_dataset,width=40,font=("times new roman",12,"bold"),bg="blue",fg="white")
        take_photo_btn.grid(row=0,column=0)

        #photo update
        update_photo_btn=Button(btn_frame2,text="Update photo",command=self.generate_dataset,width=40,font=("times new roman",12,"bold"),bg="blue",fg="white")
        update_photo_btn.grid(row=0,column=1)
               


       #right panel
        Right_frame=LabelFrame(main_frame,bd=2,bg="white",relief=RIDGE,text="Student Details",font=("times new roman",12,"bold"))
        Right_frame.place(x=780,y=10,width=660,height=580)

        #search system
        search_frame=LabelFrame(Right_frame,bd=2,bg="white",relief=RIDGE,text="Search Sytem",font=("times new roman",12,"bold"))
        search_frame.place(x=5,y=00,width=650,height=150)
        search_label=Label(search_frame,text="Search By",font=("times new roman",12,"bold"))
        search_label.grid(row=0,column=0)
        search_combo=ttk.Combobox(search_frame, font=("times new roman", 13, "bold"), state="readonly ",width=20)
        search_combo["values"]=("Select ", "USN", "Name")
        search_combo.current(0)
        search_combo.grid(row=0,column=1, padx=2, pady=1, sticky=W)

        search_entry=ttk.Entry(search_frame,width=20,font=("times new roman",13,"bold"))
        search_entry.grid(row=0,column=2,padx=10,sticky=W)

        search_btn=Button(search_frame,text="Search",width=19,font=("times new roman",12,"bold"),bg="blue",fg="white")
        search_btn.grid(row=2,column=1,pady=3)

        search_btn=Button(search_frame,text="Show all",width=19,font=("times new roman",12,"bold"),bg="blue",fg="white")
        search_btn.grid(row=3,column=1,pady=3)

        #table frame
        table_frame=LabelFrame(Right_frame,bd=2,bg="white",relief=RIDGE,font=("times new roman",12,"bold"))
        table_frame.place(x=5,y=150,width=650,height=400)

        scroll_x=ttk.Scrollbar(table_frame,orient=HORIZONTAL)
        scroll_y=ttk.Scrollbar(table_frame,orient=VERTICAL)

        self.student_table=ttk.Treeview(table_frame,column=("dept","USN","name","photo"),xscrollcommand=scroll_x.set,yscrollcommand=scroll_y.set)
        scroll_x.pack(side=BOTTOM,fill=X)
        scroll_y.pack(side=RIGHT,fill=Y)
        scroll_x.config(command=self.student_table.xview)
        scroll_y.config(command=self.student_table.yview)

        self.student_table.heading("dept",text="Department")
        self.student_table.heading("USN",text="USN")
        self.student_table.heading("name",text="Name")
        self.student_table.heading("photo",text="Photo sample satus")
        self.student_table["show"]="headings"

        
        self.student_table.pack(fill=BOTH,expand=1)
        self.student_table.bind("<ButtonRelease>",self.get_cursor)
        self.fetch_data()



    #=============function========

    def add_data(self):
        if self.var_dept.get()=="Select Department" or self.var_dept.get()=="" or self.var_USN.get()=="":
            messagebox.showerror("Error","All fields  are required",parent=self.root)
        else:
            try:
                conn=mysql.connector.connect(host="localhost",username="root",password="VK",database="face_recognition")
                my_cursor=conn.cursor()
                my_cursor.execute("Insert into student values(%s,%s,%s,%s,%s)",(     
                                                                                self.var_dept.get(),
                                                                                self.var_USN.get(),
                                                                                self.var_name.get(),
                                                                                self.var_radio1.get(),
                                                                                self.var_year.get()
                                                                        ))
                
                conn.commit()
                self.fetch_data()
                conn.close()
                messagebox.showinfo("Success","Student Data has been added",parent=self.root)
            except Exception as es:
                messagebox.showerror("Error",f"Due to :{str(es)}",parent=self.root)

    #===============fetch data==============
    def fetch_data(self):
        conn=mysql.connector.connect(host="localhost",username="root",password="VK",database="face_recognition")
        my_cursor=conn.cursor()
        my_cursor.execute("select * from student")
        data=my_cursor.fetchall()

        if len(data)!=0:
            self.student_table.delete(*self.student_table.get_children())
            for i in data:
                self.student_table.insert("",END,values=i)
            conn.commit()
        conn.close()

#========FETCH DATA============
    def get_cursor(self,event=""):
        cursor_focus=self.student_table.focus()
        content=self.student_table.item(cursor_focus)
        data=content["values"]

        self.var_dept.set(data[0]),
        self.var_USN.set(data[1]),
        self.var_name.set(data[2]),
        self.var_radio1.set(data[3]),
        self.var_year.set(data[4])

    #===update===
    def update_data(self):
        if self.var_dept.get()=="Select Department" or self.var_dept.get()=="" or self.var_USN.get()=="":
            messagebox.showerror("Error","All fields  are required",parent=self.root)
        else:
            try:        
                Update=messagebox.askyesno("Update","Do you want to update student details",parent=self.root)
                if Update>0:
                    conn=mysql.connector.connect(host="localhost",username="root",password="VK",database="face_recognition")
                    my_cursor=conn.cursor()
                    my_cursor.execute("Update student set Department=%s, Name=%s,Photosample=%s,Year=%s where USN=%s",(

                                                                                                                self.var_dept.get(),                                                                                                                
                                                                                                                self.var_name.get(),
                                                                                                                self.var_radio1.get(),
                                                                                                                self.var_year.get(),
                                                                                                                self.var_USN.get()
                                                                                                                
                                                                                                                ))
                else:
                    if not Update:
                        return
                messagebox.showinfo("Success","Stuent details updated!!",parent=self.root)
                conn.commit()
                self.fetch_data()
                conn.close()
            except Exception as es:
                messagebox.showerror("Error",f"Due To:{str(es)}",parent=self.root)
    
    #delete
    def delete_data(self):
        if self.var_USN.get()==" ":
            messagebox.showerror("Error","Student ID is required",parent=self.root)
        else:
            try:
                delete=messagebox.askyesno("Student delete page","do you want to detete this student?",parent=self.root)
                if delete>0:
                    conn=mysql.connector.connect(host="localhost",username="root",password="VK",database="face_recognition")
                    my_cursor=conn.cursor()
                    sql="delete from student where USN=%s"
                    val=(self.var_USN.get(),)
                    my_cursor.execute(sql,val)
                else:
                    if not delete:
                        return
                conn.commit()
                self.fetch_data()
                conn.close()
                messagebox.showinfo("delete","deleted succesfully",parent=self.root)
            except Exception as es:
                messagebox.showerror("Error",f"Due To:{str(es)}",parent=self.root)
    #reset
    def reset_data(self):
        self.var_dept,set("Select Department")
        self.var_USN,set("")
        self.var_name,set("")
        self.var_radio1,set("")
        self.var_year,set("Select Year")


    


    #==============generate dataset and take photo========
    def generate_dataset(self):
        if self.var_dept.get()=="Select Department" or self.var_dept.get()=="" or self.var_USN.get()=="":
            messagebox.showerror("Error","All fields  are required",parent=self.root)
        else:
            try:        
                conn=mysql.connector.connect(host="localhost",username="root",password="VK",database="face_recognition")
                my_cursor=conn.cursor()
                my_cursor.execute("select * from student")
                myresult=my_cursor.fetchall()
                id=0
                for x in myresult:
                    id=id+1
                my_cursor.execute("Update student set Department=%s,Name=%s,Photosample=%s,Year=%s where USN=%s ",(

                                                                                                                self.var_dept.get(),
                                                                                                                self.var_name.get(),
                                                                                                                self.var_radio1.get(),
                                                                                                                self.var_year.get(),
                                                                                                                self.var_USN.get()==id+1
                                                                                                                
                                                                                                                ))
                conn.commit()
                self.fetch_data()
                self.reset_data()
                conn.close()

                #======l0ad dta====
                #cap=cv2.VideoCapture(0)
                face_classifier=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
                
                def face_cropped(img):
                    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                    faces=face_classifier.detectMultiScale(gray, 1.3,5)
                    #scaling factor=1.3
                    #Minimum Neighbor=5
                    for (x,y,w,h) in faces:
                        face_cropped=img[y:y+h,x:x+w]
                        return face_cropped
                cap=cv2.VideoCapture(0)
                img_id=0
                while True:
                    ret, my_frame=cap.read()
                    if face_cropped(my_frame) is not None:
                        img_id+=1
                        face=cv2.resize(face_cropped(my_frame), (450,450))
                        face=cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                        file_name_path="data/user."+str(id)+"."+str(img_id)+".jpg"
                        cv2.imwrite(file_name_path,face)
                        cv2.putText(face, str(img_id),(50,50),cv2.FONT_HERSHEY_COMPLEX, 2, (0,255,0),2)  #store iage as user.1.01...
                        cv2.imshow("Cropped Face", face)

                    if cv2.waitKey(20)==13 or int(img_id)==100:  #to stop while pressing enter
                            break
                cap.release()
                cv2.destroyAllWindows()
                messagebox.showinfo("Result", "Generating data sets compled!!!!")
            except Exception as es:
                messagebox.showerror("Error",f"Due To:{str(es)}",parent=self.root)




                














            



       





    
        

if __name__ == "__main__":
     root=Tk()
     obj= Student(root)
     root.mainloop()
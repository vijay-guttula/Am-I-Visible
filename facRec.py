from tkinter import *
from tkinter import messagebox
import cv2
import numpy as np
import os


class FaceRecog:
    def __init__(self, root):
        self.root = root
        self.haar_file = 'haar.xml'

        # first label to add space between title bar and label 2
        Label(self.root, height=2, bg='white').pack(fill=BOTH)
        Label(self.root, bg='maroon', text='Face Recognition in Real Time', font=(
            'arial', 15), height=3, bd=2, relief=GROOVE).pack(fill=BOTH)  # 2nd lable to add the title bar

        '''
        creating a control frame to keep our buttons within.
        '''

        control_frame = Frame(self.root, height=200,
                              bg='white', bd=4, relief=RIDGE)
        control_frame.pack(fill=BOTH, pady=20, padx=10)

        """
        Now lets create the buttons
        """

        Button(control_frame, text='Train Model', bd=2, height=3, relief=GROOVE, font=(
            'arial', 12, 'bold'), width=12, command=self.get_data).place(x=50, y=50)

        Button(control_frame, text='Test Model', bd=2, height=3, relief=GROOVE, font=(
            'arial', 12, 'bold'), width=12, command=self.test).place(x=200, y=50)

        Button(control_frame, text='Exit', bd=2, height=3, relief=GROOVE, font=(
            'arial', 12, 'bold'), width=12, command=self.root.quit).place(x=350, y=50)

    def get_data(self):
        """this is to create a new window
        """
        self.top = Toplevel()
        self.top.geometry('300x200+240+200')
        self.top.configure(bg='maroon')
        self.top.resizable(0, 0)

        """this is to create a label name and its input field
        """
        name_label = Label(self.top, text='Name', width=10,
                           font=('arial', 12, 'bold')).place(x=10, y=20)
        self.name = Entry(self.top, width=15, font=('arial', 12))
        self.name.place(x=120, y=20)

        """then lets create id label and its input field
        """
        id_label = Label(self.top, text='ID', width=10, font=(
            'arial', 12, 'bold')).place(x=10, y=60)
        self.id_ent = Entry(self.top, width=15, font=(
            'arial', 12))
        self.id_ent.place(x=120, y=60)

        """
        lets create a button to it
        """
        btn = Button(self.top, text='Train model', font=(
            'arial', 12, 'bold'), command=self.train)
        btn.place(x=100, y=120)

    def train(self):
        """
        this function is to create a data set per person
        """
        name = self.name.get()
        id_ = self.id_ent.get()

        """
        if the name and id fields are not null, then the take_images function is called with
        the given name and id
        """
        if name != '' and id_ != '':
            print(name, id_)
            self.top.destroy()
            self.take_images(name, id_)
        else:
            messagebox.showwarning('Warning', 'Please fill all the fields')

    def take_images(self, name, id_):
        datasets = 'dataset'
        subdata = str(name) + '-' + str(id_)
        path = os.path.join(datasets, subdata)
        if not os.path.isdir(path):
            os.mkdir(path)

        face_cascade = cv2.CascadeClassifier(self.haar_file)

        # 0 means our primary webcam, if you use another webcam, then 1 or name
        webcam = cv2.VideoCapture(0)

        count = 1

        width, height = 130, 200

        while count <= 30:
            # stores true or false wheter the webcam is on or off and stores image.
            _, im = webcam.read()
            # image, and store it as grayscale image
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 4)

            for (x, y, w, h) in faces:
                cv2.rectangle(im, (x, y), (x+w, y + h), (255, 0, 0), 2)
                """
                croping the face
                """
                face = gray[y:y+h, x:x+w]

                face_resize = cv2.resize(face, (width, height))

                cv2.imwrite('%s/%s.png' % (path, count), face_resize)
            count += 1
            cv2.imshow('Image', im)
            key = cv2.waitKey(10)

            if key == 27:
                break

        cv2.destroyAllWindows()
        webcam.release()
        messagebox.showinfo('TrainModel', 'Data is saved')

    def test(self):
        datasets = 'dataset'
        # Create a list of images and a list of corresponding names
        (images, lables, names, id) = ([], [], {}, 0)
        for (subdirs, dirs, files) in os.walk(datasets):
            for subdir in dirs:
                names[id] = subdir
                subjectpath = os.path.join(datasets, subdir)
                for filename in os.listdir(subjectpath):
                    path = subjectpath + '/' + filename
                    lable = id
                    images.append(cv2.imread(path, 0))
                    lables.append(int(lable))
                id += 1
        (width, height) = (130, 100)

        # Create a Numpy array from the two lists above
        (images, lables) = [np.array(lis) for lis in [images, lables]]

        # OpenCV trains a model from the images
        # NOTE FOR OpenCV2: remove '.face'
        model = cv2.face.LBPHFaceRecognizer_create()
        model.train(images, lables)

        # Part 2: Use fisherRecognizer on camera stream
        face_cascade = cv2.CascadeClassifier(self.haar_file)
        webcam = cv2.VideoCapture(0)
        while True:
            (_, im) = webcam.read()
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
                face = gray[y:y + h, x:x + w]
                face_resize = cv2.resize(face, (width, height))
                # Try to recognize the face
                prediction = model.predict(face_resize)
                cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 3)

                if prediction[1] < 500:
                    cv2.putText(im, '% s - %.0f' %
                                (names[prediction[0]],
                                 prediction[1]), (x - 10, y - 10),
                                cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
                else:
                    cv2.putText(im, 'not recognized',
                                (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))

            cv2.imshow('OpenCV', im)

            key = cv2.waitKey(10)
            if key == 27:
                break
        cv2.destroyAllWindows()

    def take_images(self, name_, id_):
        # time.sleep(2)
        # All the faces data will be
        # present this folder
        datasets = 'dataset'
        # These are sub data sets of folder,
        # for my faces I've used my name you can
        # change the label here
        sub_data = str(name_) + '-' + str(id_)
        path = os.path.join(datasets, sub_data)
        if not os.path.isdir(path):
            os.mkdir(path)

        # defining the size of images
        (width, height) = (130, 100)

        # '0' is used for my webcam,
        # if you've any other camera
        # attached use '1' like this
        face_cascade = cv2.CascadeClassifier(self.haar_file)
        webcam = cv2.VideoCapture(0)

        # The program loops until it has 30 images of the face.
        count = 1
        while count < 30:
            (_, im) = webcam.read()
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 4)
            for (x, y, w, h) in faces:
                cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
                face = gray[y:y + h, x:x + w]
                face_resize = cv2.resize(face, (width, height))
                cv2.imwrite('% s/% s.png' % (path, count), face_resize)
            count += 1

            cv2.imshow('OpenCV', im)
            key = cv2.waitKey(10)
            if key == 27:
                break
        cv2.destroyAllWindows()
        messagebox.showinfo(
            "Python Says", "Model is Trained with Your \n  Image Data")


if __name__ == '__main__':
    root = Tk()
    FaceRecog(root)
    root.geometry('500x300+240+200')
    root.title('Face Recognition in Real Time')
    root.resizable(0, 0)
    root.config(bg='maroon')
    root.mainloop()

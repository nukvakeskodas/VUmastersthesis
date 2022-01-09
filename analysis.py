import pandas as pd
import numpy as np
import os
from scipy.stats import t
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import ttk

LARGE_FONT= ("Verdana", 12)

file_types = ['.dat','.txt','.csv']

files = [f for f in os.listdir('.') if os.path.isfile(f) and any(m in f for m in file_types)]

first_file, second_file = files[:2]

confidence_interval = 95

project_median = False

def data_preparation (file_name):
    iteration_file = pd.read_table(file_name, 
                                 delim_whitespace=True, 
                                 header=None)

    iteration_file = iteration_file.rename(columns={0: "iterations",
                                                    1: "value"})
    iteration_file = iteration_file[['iterations','value']]

    iteration_file['iterations'] = iteration_file['iterations'].astype(int)

    iteration_variety = sorted(iteration_file['iterations'].unique())

    #useful list
    pat = np.asarray(iteration_variety)
    N = len(pat)


    iteration_file['full_xprmnt'] = (iteration_file['iterations'].rolling(window=N , min_periods=N)
                                                                  .apply(lambda x: (x==pat).all())
                                                                  .mask(lambda x: x == 0) 
                                                                  .bfill(limit=N-1)
                                                                  .fillna(0)
                                                                  .astype(bool))

    return iteration_file[iteration_file["full_xprmnt"]==True].drop(columns=['full_xprmnt'])

def df_min_max(df):
    iterations_min_max = df.groupby(['iterations',
                                     'algorithm'])['value'].agg([max, 
                                                                 min]).stack().reset_index(level=[2]).rename(columns={'level_2': 'mm', 
                                                                                                                           0: 'values'}).reset_index()
    return iterations_min_max[iterations_min_max.mm == 'min'], iterations_min_max[iterations_min_max.mm == 'max']
        
def op_sol(df1,df2):
    op_sol1 = df1['value'].max()
    op_sol2 = df2['value'].max()
    return max(op_sol1,op_sol2)

def merge_both_tables(df1,df2):
    df1['algorithm'] = first_file
    df2['algorithm'] = second_file
    df = df1.append(df2)
    df = df.reset_index()
    return df.drop(columns='index')

def mean_sd (df):
    f, ax = plt.subplots(figsize=(20,12));

    sns.lineplot(data=df,
                 x='iterations',
                 y='value',
                 hue = 'algorithm',
                 ci='sd').axhline(op_sol,
                                  color='red',
                                  alpha=0.2);

    return f

def count_less_than(df1, df2, N):
    max_exp1 = df1.groupby('iterations').count().iloc[0][0]
    max_exp2 = df2.groupby('iterations').count().iloc[0][0]
    
    N_endings1 = df1[df1['iterations']==N]
    N_endings2 = df2[df2['iterations']==N]
    
    N_endings1['probability']=0
    N_endings2['probability']=0
    for index, row in N_endings1.iterrows():
        N_endings1.at[index,'probability'] = N_endings1[N_endings1['value']>=row['value']]['value'].count()
        
    for index, row in N_endings2.iterrows():
        N_endings2.at[index,'probability'] = N_endings2[N_endings2['value']>=row['value']]['value'].count()
        
    N_endings1['probability']=N_endings1['probability']/max_exp1
    N_endings2['probability']=N_endings2['probability']/max_exp2
    
    df = N_endings1.append(N_endings2)
    df = df.reset_index()
    return df.drop(columns='index')

def new_feature (df):
    f, ax = plt.subplots(figsize=(20,12));
    
    sns.scatterplot(data=df,
                 x='probability',
                 y='value',
                 hue = 'algorithm');
    
    return f

def confidence(df, df_min, df_max, confidence,project_median):
    g, ax = plt.subplots(figsize=(20,12));
    
    sns.lineplot(data=df,
                x='iterations',
                y='value',
                ci=confidence,
                hue='algorithm',
                alpha=0.001);
    if project_median == True:
        sns.lineplot(data=df,
                      x='iterations',
                      y='value',
                      estimator=np.median,
                     hue='algorithm',
                      ci=False,
                    legend = False);

    sns.lineplot(data=df_min,x='iterations',y='values', hue='algorithm', linestyle='--', alpha=0.6,
                legend = False);
    sns.lineplot(data=df_max,x='iterations',y='values', hue='algorithm', linestyle='--', alpha=0.6,
                legend = False);
    
    return g

def probabilities(df,op_sol,N):
    op_sol005 = op_sol*0.995
    op_sol01 = op_sol*0.99
    op_sol015 = op_sol*0.985
    op_sol02 = op_sol*0.98

    total = df[(df['iterations']==N)].count()[0]    
    probabilities0 = df[(df['iterations']==N)&(df['value']>=op_sol)].count()[0]
    probabilities005 = df[(df['iterations']==N)&(df['value']>=op_sol005)].count()[0]
    probabilities01 = df[(df['iterations']==N)&(df['value']>=op_sol01)].count()[0]
    probabilities015 = df[(df['iterations']==N)&(df['value']>=op_sol015)].count()[0]
    probabilities02 = df[(df['iterations']==N)&(df['value']>=op_sol02)].count()[0]
    
    error_probabilities = pd.DataFrame(columns=["probabilities", "error"], data=[[probabilities0,0],
                                                                                 [probabilities005,0.005],
                                                                                 [probabilities01,0.01],
                                                                                 [probabilities015,0.015],
                                                                                 [probabilities02,0.02]])
    
    error_probabilities["probabilities"] = error_probabilities["probabilities"]/total

    return error_probabilities

def probabilities_graph(df):
    f, ax = plt.subplots(figsize=(20,12));

    sns.lineplot(data=df,
                 x='error',
                 y='probabilities',
                 hue = 'algorithm',
                 ci=False).axhline(1,
                                  color='red',
                                  alpha=0.2);

    return f

iteration_file_clean1 = data_preparation(first_file)
iteration_file_clean2 = data_preparation(second_file)
iteration_file_clean = merge_both_tables(iteration_file_clean1,iteration_file_clean2)
op_sol = op_sol(iteration_file_clean1,iteration_file_clean2)
iteration_file_clean_min, iteration_file_clean_max = df_min_max(iteration_file_clean)
new_feature_data1000 = count_less_than(iteration_file_clean1, iteration_file_clean2, N=1000)
new_feature_data2000 = count_less_than(iteration_file_clean1, iteration_file_clean2, N=2000)
new_feature_data5000 = count_less_than(iteration_file_clean1, iteration_file_clean2, N=5000)
new_feature_data10000 = count_less_than(iteration_file_clean1, iteration_file_clean2, N=10000)

probs11 = probabilities(iteration_file_clean1,op_sol,N=1000)
probs21 = probabilities(iteration_file_clean2,op_sol,N=1000)
probs1000 = merge_both_tables(probs11,probs21)

probs12 = probabilities(iteration_file_clean1,op_sol,N=2000)
probs22 = probabilities(iteration_file_clean2,op_sol,N=2000)
probs2000 = merge_both_tables(probs12,probs22)

probs13 = probabilities(iteration_file_clean1,op_sol,N=5000)
probs23 = probabilities(iteration_file_clean2,op_sol,N=5000)
probs5000 = merge_both_tables(probs13,probs23)

probs14 = probabilities(iteration_file_clean1,op_sol,N=10000)
probs24 = probabilities(iteration_file_clean2,op_sol,N=10000)
probs10000 = merge_both_tables(probs14,probs24)

class SeaofBTCapp(tk.Tk):

    def __init__(self, *args, **kwargs):
        
        tk.Tk.__init__(self, *args, **kwargs)

        tk.Tk.wm_title(self, "APP")
        
        
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand = True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}

        for F in (StartPage, 
                  Mean_SD_Graph, 
                  CI_MINMAX_Graph, 
                  Probability_Graph1000, 
                  Probability_Graph2000, 
                  Probability_Graph5000, 
                  Probability_Graph10000,
                  Probability_error1000,
                  Probability_error2000,
                  Probability_error5000,
                  Probability_error10000):

            frame = F(container, self)

            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    def show_frame(self, cont):

        frame = self.frames[cont]
        frame.tkraise()
        
class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self,parent)
        label = tk.Label(self, text="Use one of the buttons to open up a graph", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        button = ttk.Button(self, text="Mean-SD",
                            command=lambda: controller.show_frame(Mean_SD_Graph))
        button.pack()

        button2 = ttk.Button(self, text="CI-Min-Max",
                            command=lambda: controller.show_frame(CI_MINMAX_Graph))
        button2.pack()

        button3 = ttk.Button(self, text="CDFR (N=1000)",
                            command=lambda: controller.show_frame(Probability_Graph1000))
        button3.pack()
        
        button4 = ttk.Button(self, text="CDFR (N=2000)",
                            command=lambda: controller.show_frame(Probability_Graph2000))
        button4.pack()
        
        button5 = ttk.Button(self, text="CDFR (N=5000)",
                            command=lambda: controller.show_frame(Probability_Graph5000))
        button5.pack()
        
        button6 = ttk.Button(self, text="CDFR (N=10000)",
                            command=lambda: controller.show_frame(Probability_Graph10000))
        button6.pack()
        
        button7 = ttk.Button(self, text="Probabilities-errors (N=1000)",
                            command=lambda: controller.show_frame(Probability_error1000))
        button7.pack()
        
        button8 = ttk.Button(self, text="Probabilities-errors (N=2000)",
                            command=lambda: controller.show_frame(Probability_error2000))
        button8.pack()
        
        button9 = ttk.Button(self, text="Probabilities-errors (N=5000)",
                            command=lambda: controller.show_frame(Probability_error5000))
        button9.pack()
        
        button10 = ttk.Button(self, text="Probabilities-errors (N=10000)",
                            command=lambda: controller.show_frame(Probability_error10000))
        button10.pack()

class Mean_SD_Graph(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Mean and Standard Deviation", font=LARGE_FONT)
        label.pack(pady=10,padx=10)
        
        button1 = ttk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage))
        button1.pack()
        
        

        fig = mean_sd(iteration_file_clean)

        canvas = FigureCanvasTkAgg(fig, self)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, self)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)


class CI_MINMAX_Graph(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Confidence Intervals with 95% and Min-Max" , font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        button1 = ttk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage))
        button1.pack()

        fig = confidence(iteration_file_clean,iteration_file_clean_min, 
                         iteration_file_clean_max, 
                         confidence=confidence_interval, 
                         project_median=project_median)

        canvas = FigureCanvasTkAgg(fig, self)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, self)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)


class Probability_Graph1000(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="CDFR, N=1000", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        button1 = ttk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage))
        button1.pack()

        fig = new_feature(new_feature_data1000)

        canvas = FigureCanvasTkAgg(fig, self)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, self)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

class Probability_Graph2000(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="CDFR, N=2000", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        button1 = ttk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage))
        button1.pack()

        fig = new_feature(new_feature_data2000)

        canvas = FigureCanvasTkAgg(fig, self)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, self)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

class Probability_Graph5000(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="CDFR, N=5000", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        button1 = ttk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage))
        button1.pack()

        fig = new_feature(new_feature_data5000)

        canvas = FigureCanvasTkAgg(fig, self)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, self)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
class Probability_Graph10000(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="CDFR, N=10000", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        button1 = ttk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage))
        button1.pack()

        fig = new_feature(new_feature_data10000)

        canvas = FigureCanvasTkAgg(fig, self)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, self)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
class Probability_error1000(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Probability to obtain optimal solution with certain error, N=1000", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        button1 = ttk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage))
        button1.pack()

        fig = probabilities_graph(probs1000)

        canvas = FigureCanvasTkAgg(fig, self)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, self)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

class Probability_error2000(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Probability to obtain optimal solution with certain error, N=2000", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        button1 = ttk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage))
        button1.pack()

        fig = probabilities_graph(probs2000)

        canvas = FigureCanvasTkAgg(fig, self)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, self)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
class Probability_error5000(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Probability to obtain optimal solution with certain error, N=5000", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        button1 = ttk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage))
        button1.pack()

        fig = probabilities_graph(probs5000)

        canvas = FigureCanvasTkAgg(fig, self)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, self)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

class Probability_error10000(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Probability to obtain optimal solution with certain error, N=10000", font=LARGE_FONT)
        label.pack(pady=10,padx=10)

        button1 = ttk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage))
        button1.pack()

        fig = probabilities_graph(probs10000)

        canvas = FigureCanvasTkAgg(fig, self)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, self)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        
app = SeaofBTCapp()
app.mainloop()
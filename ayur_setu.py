"""
Ayur-Setu: AI-Based Patient Feedback Analysis for Panchakarma Centers
ML Edition - Logistic Regression that learns from labeled data
Requirements: pip install textblob matplotlib pandas scikit-learn
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import pandas as pd
import matplotlib.pyplot as plt
import os
import csv as csv_module
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(BASE_DIR, "patient_feedback_labeled.csv")
MODEL_FILE = os.path.join(BASE_DIR, "ayursetu_model.pkl")

# ---------- ML Model ----------
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

class SentimentModel:
    def __init__(self):
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=3000, stop_words='english')),
            ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
        ])
        self.trained = False
        self.load()

    def train(self, df):
        labeled = df[df['label'].isin(['Positive','Negative','Neutral'])].dropna(subset=['feedback','label'])
        if len(labeled) < 5:
            return False
        X = labeled['feedback'].astype(str)
        y = labeled['label']
        self.pipeline.fit(X, y)
        self.trained = True
        self.save()
        return True

    def predict(self, texts):
        if not self.trained:
            return ['Unlabeled'] * len(texts)
        preds = self.pipeline.predict(texts)
        probs = self.pipeline.predict_proba(texts)
        scores = [max(p) for p in probs]
        return list(zip(preds, scores))

    def save(self):
        with open(MODEL_FILE, 'wb') as f:
            pickle.dump(self.pipeline, f)

    def load(self):
        if os.path.exists(MODEL_FILE):
            try:
                with open(MODEL_FILE, 'rb') as f:
                    self.pipeline = pickle.load(f)
                self.trained = True
            except:
                self.trained = False

model = SentimentModel()

# ---------- App ----------
class AyurSetuApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ayur-Setu -- Smart Feedback Analyzer (ML Edition)")
        self.root.geometry("1000x680")
        self.root.configure(bg="#f0f4f0")
        self.df = None
        self.build_ui()
        self.load_and_train()

    def build_ui(self):
        tk.Label(self.root, text="Ayur-Setu: AI Patient Feedback Analyzer  [ML Edition]",
                 font=("Helvetica", 15, "bold"), bg="#2e7d5e", fg="white", pady=10).pack(fill=tk.X)

        top = tk.Frame(self.root, bg="#f0f4f0", pady=6)
        top.pack(fill=tk.X, padx=10)

        btns = [
            ("Load & Analyze", self.load_and_train, "#2e7d5e"),
            ("Add Feedback",   self.open_add_feedback, "#4a90d9"),
            ("Show Chart",     self.show_chart, "#e07b39"),
            ("Retrain Model",  self.retrain, "#7b5ea7"),
        ]
        for txt, cmd, col in btns:
            tk.Button(top, text=txt, command=cmd, bg=col, fg="white",
                      font=("Helvetica", 10, "bold"), padx=8).pack(side=tk.LEFT, padx=4)

        self.train_status = tk.Label(top, text="Model: not trained", bg="#f0f4f0",
                                     fg="#888", font=("Helvetica", 9, "italic"))
        self.train_status.pack(side=tk.RIGHT, padx=10)

        main = tk.Frame(self.root, bg="#f0f4f0")
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        table_frame = tk.LabelFrame(main, text="Patient Feedbacks  (click row to view & correct label)",
                                    bg="#f0f4f0", font=("Helvetica", 10, "bold"), fg="#2e7d5e")
        table_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0,5))

        cols = ("ID", "Name", "Therapy", "Label", "Confidence")
        self.tree = ttk.Treeview(table_frame, columns=cols, show="headings", height=18)
        widths = {"ID":40, "Name":120, "Therapy":100, "Label":90, "Confidence":90}
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=widths[c])
        self.tree.tag_configure("Positive", background="#d4edda")
        self.tree.tag_configure("Negative", background="#f8d7da")
        self.tree.tag_configure("Neutral",  background="#fff3cd")
        self.tree.tag_configure("Unlabeled", background="#e8e8e8")
        self.tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.tree.bind("<ButtonRelease-1>", self.show_feedback_popup)

        # Right panel
        right = tk.Frame(main, bg="#f0f4f0")
        right.pack(side=tk.RIGHT, fill=tk.Y)

        sum_frame = tk.LabelFrame(right, text="AI Summary", bg="#f0f4f0",
                                  font=("Helvetica", 10, "bold"), fg="#2e7d5e", width=280)
        sum_frame.pack(fill=tk.BOTH, expand=True)
        sum_frame.pack_propagate(False)
        self.summary_text = scrolledtext.ScrolledText(sum_frame, font=("Courier", 9),
                                                      bg="#fffef5", wrap=tk.WORD, state=tk.DISABLED)
        self.summary_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.status = tk.Label(self.root, text="Ready.", anchor=tk.W,
                               bg="#2e7d5e", fg="white", font=("Helvetica", 9))
        self.status.pack(fill=tk.X, side=tk.BOTTOM)

    def load_and_train(self):
        if not os.path.exists(CSV_FILE):
            messagebox.showerror("Error", CSV_FILE + " not found!")
            return
        self.df = pd.read_csv(CSV_FILE)
        if 'label' not in self.df.columns:
            self.df['label'] = 'Unlabeled'

        # Train if we have labeled data
        trained = model.train(self.df)
        if trained:
            labeled_count = self.df['label'].isin(['Positive','Negative','Neutral']).sum()
            self.train_status.config(text="Model trained on " + str(labeled_count) + " samples", fg="#2e7d5e")

        # Predict for rows with no label or Unlabeled
        unlabeled_mask = ~self.df['label'].isin(['Positive','Negative','Neutral'])
        if model.trained and unlabeled_mask.any():
            texts = self.df.loc[unlabeled_mask, 'feedback'].astype(str).tolist()
            results = model.predict(texts)
            self.df.loc[unlabeled_mask, 'label'] = [r[0] for r in results]
            self.df.loc[unlabeled_mask, 'confidence'] = [r[1] for r in results]

        if 'confidence' not in self.df.columns:
            self.df['confidence'] = 0.0

        # Also predict confidence for all labeled rows if model trained
        if model.trained:
            results = model.predict(self.df['feedback'].astype(str).tolist())
            self.df['predicted'] = [r[0] for r in results]
            self.df['confidence'] = [r[1] for r in results]

        self.refresh_table()
        self.refresh_summary()
        self.status.config(text="Loaded " + str(len(self.df)) + " records.")

    def retrain(self):
        if self.df is None:
            return
        trained = model.train(self.df)
        if trained:
            labeled_count = self.df['label'].isin(['Positive','Negative','Neutral']).sum()
            self.train_status.config(text="Retrained on " + str(labeled_count) + " samples", fg="#2e7d5e")
            messagebox.showinfo("Retrained", "Model retrained on " + str(labeled_count) + " labeled samples!")
            self.load_and_train()
        else:
            messagebox.showwarning("Not enough data", "Need at least 5 labeled rows to train.")

    def refresh_table(self):
        for row in self.tree.get_children():
            self.tree.delete(row)
        for _, r in self.df.iterrows():
            conf = r.get('confidence', 0)
            conf_str = "{:.0f}%".format(float(conf)*100) if conf else "—"
            label = r.get('label', 'Unlabeled')
            tag = label if label in ('Positive','Negative','Neutral') else 'Unlabeled'
            self.tree.insert("", tk.END,
                             values=(r["id"], r["patient_name"], r["therapy_type"], label, conf_str),
                             tags=(tag,))

    def refresh_summary(self):
        df = self.df
        total = len(df)
        pos = (df['label'] == 'Positive').sum()
        neg = (df['label'] == 'Negative').sum()
        neu = (df['label'] == 'Neutral').sum()
        unlabeled = (~df['label'].isin(['Positive','Negative','Neutral'])).sum()
        labeled = pos + neg + neu
        best = df[df['label']=='Positive']['therapy_type'].mode()
        worst = df[df['label']=='Negative']['therapy_type'].mode()
        summary = (
            "THERAPY FEEDBACK SUMMARY\n"
            + "─"*38 + "\n"
            + "Total Records      : " + str(total) + "\n"
            + "Labeled            : " + str(labeled) + "\n"
            + "Unlabeled          : " + str(unlabeled) + "\n"
            + "─"*38 + "\n"
            + "Positive : " + str(pos) + " (" + str(pos*100//total if total else 0) + "%)\n"
            + "Negative : " + str(neg) + " (" + str(neg*100//total if total else 0) + "%)\n"
            + "Neutral  : " + str(neu) + " (" + str(neu*100//total if total else 0) + "%)\n"
            + "─"*38 + "\n"
            + "Best Rated Therapy : " + (best[0] if not best.empty else "N/A") + "\n"
            + "Most Complaints    : " + (worst[0] if not worst.empty else "N/A") + "\n"
            + "─"*38 + "\n"
            + "Insight: " + ("Great patient satisfaction!" if pos > neg+neu else "Needs attention.") + "\n\n"
            + "Model learns as you\ncorrect labels in popups!"
        )
        self.summary_text.config(state=tk.NORMAL)
        self.summary_text.delete("1.0", tk.END)
        self.summary_text.insert(tk.END, summary)
        self.summary_text.config(state=tk.DISABLED)

    def show_feedback_popup(self, event):
        selected = self.tree.focus()
        if not selected or self.df is None:
            return
        values = self.tree.item(selected, "values")
        if not values:
            return
        try:
            row_id = int(values[0])
        except:
            return
        rows = self.df[self.df["id"] == row_id]
        if rows.empty:
            return
        r = rows.iloc[0]
        idx = rows.index[0]

        popup = tk.Toplevel(self.root)
        popup.title("Feedback Detail")
        popup.geometry("460x360")
        popup.configure(bg="#f0f4f0")

        tk.Label(popup, text=str(r['patient_name']),
                 font=("Helvetica", 13, "bold"), bg="#2e7d5e", fg="white").pack(fill=tk.X)
        tk.Label(popup,
                 text="Therapy: " + str(r['therapy_type']) + "   |   Current Label: " + str(r.get('label','?')),
                 font=("Helvetica", 10), bg="#f0f4f0", fg="#333").pack(pady=5)

        tk.Label(popup, text="Patient Feedback:", font=("Helvetica", 10, "bold"),
                 bg="#f0f4f0", anchor=tk.W).pack(fill=tk.X, padx=12)
        txt = scrolledtext.ScrolledText(popup, font=("Helvetica", 10), bg="#fffef5",
                                        wrap=tk.WORD, height=6)
        txt.pack(fill=tk.BOTH, expand=True, padx=12, pady=4)
        txt.insert(tk.END, str(r['feedback']))
        txt.config(state=tk.DISABLED)

        # Correct label section
        tk.Label(popup, text="Correct the label if wrong:",
                 font=("Helvetica", 10, "bold"), bg="#f0f4f0").pack(pady=(6,2))

        btn_frame = tk.Frame(popup, bg="#f0f4f0")
        btn_frame.pack()

        def correct(new_label):
            self.df.at[idx, 'label'] = new_label
            # Save correction back to CSV
            self.df.to_csv(CSV_FILE, index=False, quoting=csv_module.QUOTE_ALL)
            # Retrain model with updated labels
            model.train(self.df)
            labeled_count = self.df['label'].isin(['Positive','Negative','Neutral']).sum()
            self.train_status.config(
                text="Retrained on " + str(labeled_count) + " samples", fg="#2e7d5e")
            self.refresh_table()
            self.refresh_summary()
            self.status.config(text="Label corrected to '" + new_label + "' and model retrained.")
            popup.destroy()

        label_colors = {"Positive":"#2e7d5e", "Neutral":"#e0a800", "Negative":"#c0392b"}
        for lbl, col in label_colors.items():
            tk.Button(btn_frame, text=lbl, command=lambda l=lbl: correct(l),
                      bg=col, fg="white", font=("Helvetica", 10, "bold"),
                      width=10).pack(side=tk.LEFT, padx=6)

        def delete_entry():
            if messagebox.askyesno("Delete", "Are you sure you want to delete this feedback?", parent=popup):
                self.df.drop(index=idx, inplace=True)
                self.df.reset_index(drop=True, inplace=True)
                self.df.to_csv(CSV_FILE, index=False, quoting=csv_module.QUOTE_ALL)
                self.refresh_table()
                self.refresh_summary()
                self.status.config(text="Feedback deleted.")
                popup.destroy()

        tk.Button(popup, text="Delete This Entry", command=delete_entry,
                  bg="#c0392b", fg="white", font=("Helvetica", 10, "bold")).pack(pady=(0, 4))

        tk.Button(popup, text="Close", command=popup.destroy,
                  bg="#555", fg="white", font=("Helvetica", 10)).pack(pady=(0, 8))

    def show_chart(self):
        if self.df is None:
            messagebox.showwarning("No Data", "Load data first.")
            return
        counts = self.df['label'].value_counts()
        counts = counts[counts.index.isin(['Positive','Negative','Neutral'])]
        colors = {"Positive":"#4CAF50","Negative":"#f44336","Neutral":"#FFC107"}

        fig, axes = plt.subplots(1, 3, figsize=(13,4))
        fig.suptitle("Ayur-Setu Sentiment Dashboard", fontsize=13, fontweight="bold")

        # Bar
        bars = axes[0].bar(counts.index, counts.values,
                           color=[colors.get(s,"gray") for s in counts.index])
        axes[0].set_title("Sentiment Distribution")
        for bar, val in zip(bars, counts.values):
            axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                         str(val), ha="center", fontweight="bold")

        # Pie
        axes[1].pie(counts.values, labels=counts.index, autopct="%1.0f%%",
                    colors=[colors.get(s,"gray") for s in counts.index], startangle=90)
        axes[1].set_title("Sentiment Share")

        # Therapy-wise breakdown
        therapy_sentiment = self.df[self.df['label'].isin(['Positive','Negative','Neutral'])]
        therapy_counts = therapy_sentiment.groupby(['therapy_type','label']).size().unstack(fill_value=0)
        therapy_counts.plot(kind='bar', ax=axes[2],
                            color=[colors.get(c,"gray") for c in therapy_counts.columns])
        axes[2].set_title("Therapy-wise Sentiment")
        axes[2].set_xlabel("")
        axes[2].tick_params(axis='x', rotation=30)

        plt.tight_layout()
        plt.show()

    def open_add_feedback(self):
        win = tk.Toplevel(self.root)
        win.title("Add Patient Feedback")
        win.geometry("440x400")
        win.configure(bg="#f0f4f0")

        fields = [("Patient Name","name"), ("Therapy Type","therapy"), ("Feedback","fb")]
        entries = {}
        for label, key in fields:
            tk.Label(win, text=label, bg="#f0f4f0", font=("Helvetica",10)).pack(anchor=tk.W, padx=15, pady=(8,0))
            if key == "fb":
                e = tk.Text(win, height=5, font=("Helvetica",10))
            else:
                e = tk.Entry(win, font=("Helvetica",10), width=35)
            e.pack(padx=15, fill=tk.X)
            entries[key] = e

        # Manual label option
        tk.Label(win, text="Your Label (optional — model will predict if left as Auto):",
                 bg="#f0f4f0", font=("Helvetica", 9)).pack(anchor=tk.W, padx=15, pady=(8,0))
        label_var = tk.StringVar(value="Auto")
        lf = tk.Frame(win, bg="#f0f4f0")
        lf.pack(anchor=tk.W, padx=15)
        for opt in ["Auto","Positive","Neutral","Negative"]:
            tk.Radiobutton(lf, text=opt, variable=label_var, value=opt,
                           bg="#f0f4f0", font=("Helvetica",10)).pack(side=tk.LEFT, padx=4)

        def save():
            name = entries["name"].get().strip()
            therapy = entries["therapy"].get().strip()
            fb = entries["fb"].get("1.0", tk.END).strip()
            if not (name and therapy and fb):
                messagebox.showwarning("Missing", "Please fill all fields.", parent=win)
                return

            chosen_label = label_var.get()
            if chosen_label == "Auto":
                if model.trained:
                    result = model.predict([fb])
                    chosen_label = result[0][0]
                else:
                    chosen_label = "Unlabeled"

            # Get next id
            next_id = 1
            if os.path.exists(CSV_FILE):
                try:
                    existing = pd.read_csv(CSV_FILE)
                    if not existing.empty:
                        next_id = int(existing["id"].max()) + 1
                except:
                    next_id = 1

            file_exists = os.path.exists(CSV_FILE)
            with open(CSV_FILE, "a", newline="") as f:
                writer = csv_module.writer(f, quoting=csv_module.QUOTE_ALL)
                if not file_exists:
                    writer.writerow(["id","patient_name","therapy_type","feedback","label"])
                writer.writerow([next_id, name, therapy, fb, chosen_label])

            # Retrain if manually labeled
            if label_var.get() != "Auto":
                self.df = pd.read_csv(CSV_FILE)
                model.train(self.df)

            win.destroy()
            self.load_and_train()

        tk.Button(win, text="Save & Analyze", command=save,
                  bg="#2e7d5e", fg="white", font=("Helvetica",11,"bold")).pack(pady=12)


if __name__ == "__main__":
    root = tk.Tk()
    app = AyurSetuApp(root)
    root.mainloop()
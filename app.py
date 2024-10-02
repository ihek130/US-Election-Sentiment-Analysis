from flask import Flask, render_template, request  
from nltk.sentiment.vader import SentimentIntensityAnalyzer # type: ignore
import matplotlib.pyplot as plt # type: ignore
import io
import base64

app = Flask(__name__)
sid = SentimentIntensityAnalyzer()

@app.route('/', methods=['GET', 'POST'])
def index():
    sentiment = None
    score = {}
    tweet = ''
    color = 'black'
    plot_url_bar = None
    plot_url_pie = None
    
    if request.method == 'POST':
        tweet = request.form['tweet']
        score = sid.polarity_scores(tweet)
        compound = score['compound']
        
        if compound >= 0.05:
            sentiment = 'Positive'
            color = 'green'
        elif compound <= -0.05:
            sentiment = 'Negative'
            color = 'lightcoral'
        else:
            sentiment = 'Neutral'
            color = 'yellow'
        
        # Generate bar chart
        img_bar = io.BytesIO()
        labels_bar = ['Positive', 'Neutral', 'Negative']
        sizes_bar = [score['pos'], score['neu'], score['neg']]
        colors_bar = ['green', 'grey', 'red']
        plt.figure(figsize=(6, 4))
        plt.bar(labels_bar, sizes_bar, color=colors_bar)
        plt.title('Sentiment Analysis (Bar Chart)')
        plt.xlabel('Sentiment')
        plt.ylabel('Scores')
        plt.savefig(img_bar, format='png')
        img_bar.seek(0)
        plot_url_bar = base64.b64encode(img_bar.getvalue()).decode()
        img_bar.close()
        
        # Generate pie chart
        img_pie = io.BytesIO()
        labels_pie = ['Positive', 'Neutral', 'Negative']
        sizes_pie = [score['pos'], score['neu'], score['neg']]
        colors_pie = ['green', 'yellow', 'red']
        plt.figure(figsize=(6, 4))
        plt.pie(sizes_pie, labels=labels_pie, colors=colors_pie, autopct='%1.1f%%', startangle=140)
        plt.title('Sentiment Analysis (Pie Chart)')
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.savefig(img_pie, format='png')
        img_pie.seek(0)
        plot_url_pie = base64.b64encode(img_pie.getvalue()).decode()
        img_pie.close()

        return render_template('index.html', sentiment=sentiment, score=score, tweet=tweet, plot_url_bar=plot_url_bar, plot_url_pie=plot_url_pie, color=color)
    
    return render_template('index.html', sentiment=sentiment, score=score, tweet=tweet, color=color)

if __name__ == '__main__':
    app.run(debug=True)

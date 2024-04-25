# Company name generator

The goal of this project is to learn a new technology in a week. I want to learn the basics of text generation with a Recurrant Neural Network. The project will take 1 week from planning to deployment. 

> Win Case: At the click of a button I can generate a one or two word name for my new company
> 

I am using this day-by-day planner to keep track of the tasks required to complete the project and achieve the win case.

[Project Schedule](https://www.notion.so/Project-Schedule-a6d9a7b44e894d60b171553f393d5cec?pvs=21)

## The End Result

While I don't think it will be naming anyone's children anytime soon, it clearly has begun to recognize the structure and patterns of company names. To it's credit, it made multiple names that made me laugh and a few that I might use for future projects. This functionality, although limited, paired with my now much more comprehensive understanding of RNNs has made this project a tremendous success in my mind. 

[Company%20name%20generator%20e1bcef8f7aa648eeb667b0b7048e412d/nameGenVid3.mov](Company%20name%20generator%20e1bcef8f7aa648eeb667b0b7048e412d/nameGenVid3.mov)

I was unable to host the app for free due to the size of the files required to run the dashboard but there is a quickstart guide at the top of the [README.md](http://readme.md) in github that can give you access to the dashboard with minimal setup.

### Things I would do if I had another week:

- [ ]  generate a random sequence of characters and generate on that instead of user defined input
- [ ]  have a way of scoring the resulting names against the training data.
    
    This would be a sort of metric to measure the model's understanding of the training data
    
- [ ]  Add a tagline to each generated company name

### Technologies used:

- For the project's file structure and general organization:
    
    [drivendata/cookiecutter-data-science](https://github.com/drivendata/cookiecutter-data-science)
    
- For the machine learning aspect of the project I used Tensorflow and generally followed this guide:
    
    [Text generation with an RNN | TensorFlow Core](https://www.tensorflow.org/tutorials/text/text_generation#process_the_text)
    
- For the product Dashboard and UI, I used Streamlit:
    
    [Streamlit- Deploy a Machine Learning Model without learning any web framework.](https://towardsdatascience.com/streamlit-deploy-a-machine-learning-model-without-learning-any-web-framework-e8fb86079c61)
    
- For planning, note-taking and task tracking, I used Notion:
    
    [Notion - The all-in-one workspace for your notes, tasks, wikis, and databases.](https://www.googleadservices.com/pagead/aclk?sa=L&ai=DChcSEwjzmurR-dnpAhVOvsAKHZ1MDEkYABAAGgJpbQ&ohost=www.google.com&cid=CAESQeD2P5IAsEYMEdpCjNxp5Oy_IPMM-2q4ql4pmCdj8h8CvORBIJ95quMeUcGBILZK8iWnCXF3IeAv_5oSBBtYPFiD&sig=AOD64_0ybBTZOezSjJjYam8aeZMy3gMDIg&q=&ved=2ahUKEwiAjOHR-dnpAhWQVc0KHZnJBHAQ0Qx6BAgfEAE&adurl=)
    

# Day 1: Data Processing and building the model

- Scraping training data from the web with Beautiful Soup
    
    As a jumping off point I am starting with the list of Fortune 1000 companies because if you name a company based on already successful companies you cant fail, right? Using Python's requests library and beautiful soup I am going to scrape the names from the table on a web page (found here: '[https://cyber.harvard.edu/archived_content/people/edelman/fortune-registrars/fortune-list.html](https://cyber.harvard.edu/archived_content/people/edelman/fortune-registrars/fortune-list.html)'). My scrape_names() is shown below. The return value is a list of company names as strings.
    
    ```python
    def scrape_names(url, tableColumnIndex): # This function could be expanded to to accept a column name and then find the index of that column itself
        r = requests.get(url)
        soup = BeautifulSoup(r.text, 'html.parser')
        
        # find the table and extract the desired column
        tableColumn = []
        for item in soup.find_all('tr'): # find all table rows in the page
            td = item.find_all('td')[tableColumnIndex].get_text() # The "1" here will correspond to the column of the data we want to collect
            tableColumn.append(td)
        
        return tableColumn
    ```
    
    This function could be expanded in the following ways:
    
    - accept a columnName parameter and then find the index itself instead of being told the index explicitly
    - Search through multiple tables on a page to find the columnName parameter
    - Return the result set in a variety of formats
- Preprocessing our data
    
    In the data preparation steps (The 4 steps under the "Process the text" subheading) I am noticing that company names are being broken up into incomplete fragments. 
    
    > 'Wal-Mart St'
    'ores Exxon'
    ' Mobil Gen'
    'eral Motors'
    ' Ford Moto'
    > 
    
    It is my intuition that telling an RNN that "Wal-Mart St" is an acceptable name, will result in gibberish name generation. I think that making sequences out of each actual name and then padding those sequences might give us more realistic predictions.
    
    Ok, Here is what I came up with... The tutorial would have you split the text into items of length = seq_length which is an arbitrary value, and then tokenize those items. Instead, I chose to loop through the `names` list and tokenize each element using the `char2idx` mapping object we created earlier. That looks like this:
    
    ```python
    raw_input = []
    for name in names:
        raw_input.append([char2idx[c] for c in name])
    ```
    
    Then I call `pad_sequences()` with `padding='post'`, `truncating='post'`, and `maxlen=15`. This will ensure that the encoding of the name will start on the 0th index of the sequence, longer names will have the tail truncated and no names will be longer than 15 characters. Exxon Mobil = `[14, 59, 59, 50, 49, 0, 22, 50, 38, 45, 47, 0, 0, 0, 0]` From here on out we will use `padded_inputs` instead of `text_as_int` . 
    
    Examine our `char_dataset` with the following code:
    
    ```python
    char_dataset = tf.data.Dataset.from_tensor_slices(padded_inputs)
    
    # examine
    for i in char_dataset.take(2):
        print("".join(idx2char[i.numpy()]))
    
    for s in char_dataset.take(2):
        print(s) # prints the first 2 tensors in the char_dataset which are "Wal-Mart Stores" and "Exxon Mobil"
    ```
    
    Output: 
    
    ```
    Wal-Mart Stores
    Exxon Mobil    
    tf.Tensor([32 37 47  5 22 37 53 55  0 28 55 50 53 41 54], shape=(15,), dtype=int32)
    tf.Tensor([14 59 59 50 49  0 22 50 38 45 47  0  0  0  0], shape=(15,), dtype=int32)
    ```
    
    Another necessary change to the tutorial has to be made. The tutorial uses the `batch()` method to convert the long string of shakespearean text into sequences of length = `seq_length`. That sequence will then be shifted to create inputs and targets. However we already have our sequences as elements in the char_dataset so we will do
    
     `dataset = char_dataset.map(split_input_target)`
    
    instead of 
    
    `dataset = sequences.map(split_input_target)`
    
    Printing our inputs and targets should give us: 
    
    ```
    input data: 'Wal-Mart Store'
    target data: 'al-Mart Stores'
    input data: 'Exxon Mobil   '
    target data: 'xxon Mobil    '
    input data: 'General Motors'
    target data: 'eneral Motors '
    
    ```
    
- Build our Model v1
    
    All that is left is to build and train the model, these steps are the same as in the tutorial. When we try the model we get some encouraging information, An untrained model predicts `z-ilTGIHKH'I`A` to be the next character. This is encouraging because even though it is gibberish, it looks to be about the right length and consistency of what we want our answers to look like. I built the same model as they have in the tutorial with the following default params 
    
    `vocab_size = len(vocab)`
    
    `embedding_dim = 256`
    
    `rnn_units = 1024`
    
    After training the model for 100 epochs, I ran the same code that I did earlier to produce "`ilTGIHKH'I`A`", the results this time were remarkably better. 
    
    ```python
    for input_example_batch, target_example_batch in dataset.take(1):
        ex_batch_predictions = model(input_example_batch)
        print(ex_batch_predictions, "# (batch_size, sequence_length, vocab_size)")
    
    sampled_indicies = tf.random.categorical(ex_batch_predictions[0], num_samples=1)
    sampled_indicies = tf.squeeze(sampled_indicies, axis=1).numpy()
    ```
    
    Decoding the returned `sampled_indicies` yielded the model's first name, ".ooied Western". While it's not perfect, the network clearly understands the need to capitalize the first letter after a space. Even though the generation process is a little rough and not exactly what we are looking, for I am happy to call it a day and leave additional testing and tuning for tomorrow.
    
    And with that, Day 1 is over.
    
    - All that is left is to build and train the model, these steps are the same as in the tutorial. When we try the model we get some encouraging information, An untrained model predicts `z-ilTGIHKH'I`A` to be the next character. This is encouraging because even though it is gibberish, it looks to be about the right length and consistency of what we want our answers to look like. Lets train our model...

# Day 2: Tuning, Data Expansion and Dashboard Design

- Lets do some tuning...
    
    After switching the batch size to 1 and just 10 epochs of training, the model was producing names like `' MU X leli   a'` based on the random input `[[31  8 33 10  9 19 37 26 50 13 10 12 55 25]]` . 
    
    After increasing the epochs to 50, the resulting names were equally as disappointing, producing even more gibberish. Something important to note about this adjustment is that the loss started increasing from .7571 after just 10 epochs so there is definitely some overfitting when training for 50 epochs. Here are some examples for your own entertainment
    
    > '. uaiSeloToo '
    'ueon hP o C'
    'ueRi eAnlwoe'
    'oi lI.S T '
    ' Tlee eaaiBTd'
    > 
    
    After decreasing the training epochs to 15 and adding the optional inputText parameter to the `generate_name()` function the results weren't much better. These were created with input="Ethan"
    
    > "mrmn'"
    'mmela'
    'iea d'
    'BeTl '
    'llel '
    > 
    
    Something I realized is that I set the batches to 1 when I ran `batches = char_dataset.batch(64, drop_remainder=True)` earlier on. Returning this to 64 had an interesting effect. The loss in training started much higher, 2.248, and then leveled off around the same .75 as the other configurations. When generating a name with unseen data as input, `input="Ethan"`, the results were slightly better and even included an anagram:
    
    > 'leird'
    'ahor '
    'BmomB'
    > 
    
    But when I used a name from the training data as input, `input='Exxon Mobil'`, the results were **much** better. It's obvious that the network does much better on data it has seen before which leads me to believe it needs a larger training dataset.
    
    > 'n on Cobil '
    'Oper Mobil '
    'ceensMebil '
    > 
    
    Changing the GRU layer to a bidirectional layer yielded a loss metric that started at only .919 and decreased to .056 over 10 epochs. Generating a name with text as input returns almost identical names and removing that input returns more gibberish.
    
    Removing the Bidirectional GRU resulted in some more encouraging results when using the Tutorial's `generate_text()` method. Occasionally the model would predict a capitalized word like "Energy" or "Bancorp" or "Motors" or something on top of the start string which is "th" in this case.
    
    > thon Bancorp
    theal Motors
    thelley Stre
    thellet
    > 
    
    Adjusting the temperature setting on the generate text function allows me to generate more exciting names including one that looks like Texas Instruments. I would go on to find a number of names that are very similar to existing companies.
    
    > PenMark Information
    Perex Holdings 
    tiTexas Industree
    Tilwell Financial
    Glens Thole Foods
    > 
    
    With names like these I am comfortable with where the model is on it's predictions. I am going to consider the tuning done for now and cross this off the list. Here are some things that Imight try if I wanted to continue tuning.
    
    - [ ]  Use an LSTM instead of the GRU, try both unidirectional and bidirectional
    - [ ]  Make the GRU Bidirectional
    - [ ]  Experiment with additional Dense layers
    - [ ]  Adjust embedding dimension and rnn_units for each layer
    - [ ]  Adding an additional GRU layer
- Additional data gathering
    
    Generating company names based on the Fortune 1000 companies is great and all but I want to expand this model's capabilities to other sets of data too. I have a couple of ideas for new data sets, one is baby names and the other is planet names. Lets get some data sets for each of these and then train the model on those to see how it does. The following code scrapes the top 1000 baby boy names from a website.
    
    ```python
    url = 'https://www.whattoexpect.com/baby-names/list/top-baby-names-for-boys/'
    # url = 'https://www.whattoexpect.com/baby-names/list/top-baby-names-for-girls/'
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')
    listItems = soup.find_all('li')
    # for i in range(len(listItems)):
    #     if 'Liam' in listItems[i]:
    #         print(i)
    babyNames = []
    for name in listItems[65:-22]:
        babyNames.append(name.text)
    ```
    
    This definitely created some interesting names for your child, I found "ethaan" pretty quickly but there is a lot of garbage.
    
    > axton
    aclison
    bitton
    bLaham
    > 
    
    Now lets try training on a list of boy and girl names to see what we get. Training time is significantly longer but it is worth the wait because these names are much more acceptable.
    
    > bella
    bert
    minna
    masen
    > 
    
    These are just a few of the ones that stood out while I generated some names. I am not sure what the goal was here because anything that isn't a recognizable name already sounds kind of ridiculous. The company names sound much better. 
    
- Design a dashboard for the model to live on
    
    My immediate thought is to build a dashboard using Dash and flask and to put inside a docker container that can run on my home server and be connected to a domain name. I'd like to have a text box for users to type in and then as they type predicted names will be displayed beneath the text box. I would also like to include a slider for the temperature setting of the `generate_text()` method. Another important feature will be a dropdown menu allowing users to select the training data that the model uses. This will allow for expansion into things like baby names or planet names. Lastly I will need a visualization because you cant have a dsahboard without one. For this I will use Jianzheng Liu's Python library found here: 
    
    [jzliu-100/visualize-neural-network](https://github.com/jzliu-100/visualize-neural-network)
    
    Thats the design for the dashboard and with that done it is time to end Day 2.
    

# Day 3: Building the Dashboard

Today's focus is the dashboard that the a model will live on. This is important because the model may as well not exist if it can't function and be shared over the internet. I am going to use the Streamlit library because it seems to be popular in the DS and ML communities for it's simplicity and ease of development compared to other web frameworks. Working off the design from yesterday I'll start by creating all of the elements that will be on the page and then work on connecting the model to those components. Using this wonderful guide and the provided Github repository as a guide I made my own dashboard. 

[Streamlit- Deploy a Machine Learning Model without learning any web framework.](https://towardsdatascience.com/streamlit-deploy-a-machine-learning-model-without-learning-any-web-framework-e8fb86079c61)

It is only once I ran my dashboard that I discovered that my training checkpoint files were being saved in Google Collab instead of on my local filesystem making it impossible to load the model from those checkpoint files. To solve this I manually downloaded the checkpoint files from the last training epoch and dragged them into my project directory where the Streamlit app can access them. Success! After 5 hours the dashboard is finally operational. To fix the downaloading issue, I manually downloaded the checkpoint, ckpt_10.index, ckpt_10.data-00000-of-00002 and ckpt_10.data-00001-of-00002 files and placed them in my app's checkpoint directory. Next I added `.expect_partial()` to `model.load_weights(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()` in the `load_model_from_checkpoint()` function.

# Day 4: Result Filtering and Deploying the Dashboard

Today's focus is on polishing the model in the context of the dashboard and wrapping up the project. 

- Result filtering
    
     I am adding a result filtering check to the `generate_text()` call. The result filtering currently checks two conditions, 
    
    1. the name we generate in the current step is not identical to the starting string
    2. the name we generate has not already been generated
    
    When these two conditions evaluate to `True`, we need to generate a new name until they are both `False`. The code for this is below along with a step-by-step explanation.
    
    ```python
    generatedNames = [] # establish empty variables
    genName = ''        # ^^^^
    cond1 = (genName == start_string) # Condition 1
    for i in range(int(numNames)):
        genName = generate_text(model, start_string, temp, char2idx, idx2char)
        cond2 = (genName in generatedNames) 
        if cond1 or cond2:
            print('{} == {}'.format(genName, start_string))
            while cond1 or cond2:
                print('Trying again...')
                genName = generate_text(model, start_string, temp, char2idx, idx2char)
                cond2 = (genName in generatedNames) 
                print('Now this: {} ?= {}'.format(genName, start_string))
        # st.write('{}: {}'.format(i+1, genName))
        generatedNames.append(genName)
    st.write(generatedNames)
    ```
    
    - Where `numNames` is the value of the dropdown in the sidebar, either 5, 10 or 15.
    - Generate a name with the start sting as input.
    - Evaluate the truthiness of condition 2, is the name we generated in our list of names
    - If the name generated is identical to start string `or` it has been generated already:
        - while this is the case, generate a new name and check the truthiness again
    - Once we get a name that evaluates both conditions to `False` we can add it to the list and write the list to the app
- Deploying the Streamlit app
    
    After a little bit of googling it would appear that the only free option is to deploy the streamlit app on Heroku's free tier. Here is the forum post that has led me to this decision, 
    
    [Hosting streamlit on github pages](https://discuss.streamlit.io/t/hosting-streamlit-on-github-pages/356/18)
    
    The process for doing this is as follows:
    
    Create a virtual environment and install all the packages you need.
    
    `python3 -m venv env` where env is the name of the virtual environment
    
    Activate your virtual environment with  `source env/bin/activate`
    
    Go through each of your required packages (the packages you import at the top of your app) and pip install them
    
    Save your virtual environment configuration to a text file with `pip freeze > requirements.txt`
    
    Now we will create our [setup.sh](http://setup.sh) file. For this you will need a Heroku account. Once you have one create your setup.sh file with this:
    
    ```bash
    echo "\
    [general]\n\
    email=\"your-email@domain.com\"\n\
    " > ~/.streamlit/credentials.toml
    
    echo "\
    [server]\n\
    enableCORS=false\n\
    headless = true\n\
    port = $PORT\n\
    " > ~/.streamlit/config.toml
    ```
    
    Now you need a Heroku Procfile. name a new file Procfile and put this in it where [`app.py`](http://app.py) is the name of your streamlit app:
    
    ```bash
    web: sh setup.sh && streamlit run app.py
    ```
    
    Now you have all the necessary files to get your project deployed. Next I am going to put the project on Github but your project can also be deployed with the Heroku CLI.
    
    Once you have your files on Github you can connect your GH account to Heroku and select the repository you want to deploy. I had trouble committing my dashboard folder to github so I had to create a new branch and manually upload the app.py, requirements.txt, procfile and [setup.sh](http://setup.sh) to that branch. Once I did that I selected the dashboard branch from the Heroku app dashboard and deployed the app. Or at least I should have, Because my project's required files were 700MB, well over Heroku's 500MB limit, the app can not be hosted for free. Perhaps I will find another solution or just shell out the cash in the future but for now I am compromising for a Quickstart guide in the Github [README.md](http://readme.md) in case anyone wants to try it out themselves
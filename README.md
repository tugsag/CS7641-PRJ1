# CS7641-PRJ1
Repo for project 1 of CS7641

To run the script, simply run from the console: 'python main.py'. A list of options will appear that looks like the following:

Choose algorithm:
                 Decision tree: DT
                 SVM: SVM
                 Boosting: B
                 KNN: KNN
                 ANN: ANN 
                 
Enter anything other than the given choices to exit without running the algorithms. 
After choosing the algorithm, another option to choose the dataset will appear:

Choose dataset:
                epilepsy: E
                pulsar: P 
                
Once both are chosen, the model will run and print out the necessary statistics. It will also save the graphics to the 'figures' subdirectory. Some algorithms like boosting might take a while to run, so it's important to not interrupt it.

The stock data files for both datasets are included in this repo. The script will automatically process the data each time it is run. 

Only the conventional machine learning libraries are required as follows:
  pandas 1.1.1
  numpy 1.18.8
  scikit-learn 0.23.2
  matplotlib 3.1.3
  
  

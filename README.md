# PAMAE: Parallel k-Medoids Clustering with High Accuracy and Efficiency

Welcome to the PAMAE repository! This project is created by [Francesco Dal Bello](https://github.com/francescodalbi) and contains the source code for the PAMAE application.

## Description

PAMAE is the python implementation of the algorithm described in this [paper](https://dl.acm.org/doi/10.1145/3097983.3098098) and has the following functionality
I merely implemented it in python, the intellectual rights and merits are of the researchers cited.
<img src="https://github.com/francescodalbi/pamae/assets/32592051/6f78c5e4-08b4-4816-8b3c-53393a4edd5d" width="650" height="400">

## Libraries
<img src="https://github.com/francescodalbi/pamae/assets/32592051/2572d706-b268-4993-997b-bc13752f9200" width="650" height="400">



## Configuration
```
  pamae(ds_import, 2, 1000, 5)
```
  To use the algorithm change the parameters in the pamae() function --> pamae(dataset, number_of_sample, sample_size, number_of_clusters).
  The dataset should be in csv format and should contain only quantitative values, so any pre-editing of the file may be necessary.
  For plots to make sense the data must be in two dimensions, loading data in p dimensions the plots will be equally generated but meaningless.

## Output example
**Phase 1 output:** ![image](https://github.com/francescodalbi/pamae/assets/32592051/f85bb12c-44e8-49d4-98b8-0eaa6b6d8bd3 )


**Phase 2 output:** ![image](https://github.com/francescodalbi/pamae/assets/32592051/7582b5eb-fd79-418b-8353-76fe2bc57236)

## MongoDB export
The algorithm saves the result of the processing in a MongoDB database
![image](https://github.com/francescodalbi/pamae/assets/32592051/45696098-1a92-47ed-ac3e-040869a9ef4e)
![image](https://github.com/francescodalbi/pamae/assets/32592051/6f454faa-3a34-42f1-b7dc-97d795963485)


This is the result on a benchmark dataset
![image](https://github.com/francescodalbi/pamae/assets/32592051/f6cf86f8-6a1d-4d66-bb0e-2b41d1e6d045)

## Contributions

Thank you for the suggestions provided: [Pierluigi](https://github.com/zACIID) and [Filippo](https://github.com/filippodaminato)

## License

This project is licensed under the MIT Licence. See the `LICENSE` file for more details.

## Contact

If you have any questions or suggestions regarding Pamae, please feel free to contact us:

- Email: dalbellofrancesco.00@gmail.com
- GitHub: [francescodalbi](https://github.com/francescodalbi)

I appreciate your interest in my implementation of PAMAE and look forward to your feedback!


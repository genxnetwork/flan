# Federated Learning ANcestry

This Python package provides a robust and efficient solution to predict an individual's genetic ancestry using genomic data. The tool leverages federated learning to train a single model on separate datasets from different nodes.

## Features
- Federated Prediction of Ancestry: It can train a federated model using separated datasets on the different nodes. Individual genotypes are not shared between nodes.   
- Global Ancestry Prediction: It uses global ancestry data to provide a broad overview of an individual's ancestral roots.
- High Performance: Designed to handle large genomic datasets effectively.
- User-Friendly: It provides a straightforward command-line interface, making it easy to use for both bioinformaticians and geneticists.

## Installation

You can install the Ancestry Prediction tool using poetry. Make sure that you have Python 3.9 or later and poetry installed on your system.

```
poetry add https://github.com/genxnetwork/flan/
```

## Usage

To fit the model globally, use the following command:

```flan global fit```

After fitting the model, you can predict the global ancestry using the following command. Replace `file.vcf.gz` with the path to your own vcf file.

```flan global predict --file=file.vcf.gz```

To fit the model using the client-server federated architecture, you need to launch server which will perform the aggregation. Server should be accessible from all client nodes. 

Server side:
```flan server fit --server.host=127.0.0.1 --server.port=57632```

Then, you should launch clients on each node. 

Client side:
```flan client fit --server.host=127.0.0.1 --server.port=57632```

You can then predict the ancestry on the client side using the federated model and the following command:

```flan client predict --file=file.vcf.gz```

## Citation

If you use `flan` in your research, please cite it as follows:

```
TO BE ADDED BY THE PACKAGE AUTHOR
```

## Contribute

Contributions to improve this tool are welcome! You can contribute in several ways:

1. Report bugs or suggest new features by opening an issue on our GitHub page.
2. Improve the code: If you have a bug fix or a new feature you'd like to add, please open a pull request.
3. Improve the documentation: If you find any mistake or ambiguity in the documentation, please make a pull request to help us improve it.

import os
from dataclasses import dataclass
from pathlib import Path
import gc
from typing import Callable, Dict, List, Optional, Tuple, Union
from flwr.common import FitRes, MetricsAggregationFn, NDArrays, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy

import numpy 
import pandas
import scipy.sparse.linalg as linalg
from flwr.server.strategy import FedAvg
from flwr.server import start_server, ServerConfig
from flwr.client import NumPyClient
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
    ndarrays_to_parameters
)

from ..utils.plink import run_plink
from ..utils.cache import FileCache, CacheArgs
from ..fl_engine.utils import ClientArgs, ServerArgs, NodeArgs, run_node


class AlleleFreqStrategy(FedAvg):
    def __init__(self, freq_path: Path, **kwargs) -> None:
        super().__init__(**kwargs)
        self.freq_path = freq_path
        
    def aggregate_fit(self, 
                      server_round: int, 
                      results: List[Tuple[ClientProxy, FitRes]], 
                      failures: List[Tuple[ClientProxy, FitRes]]) -> Tuple[Parameters, Dict[str, Scalar]]:
        
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}
        
        # Convert results
        frequencies = [
            parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results
        ]
        
        alt_counts = sum([freq[0] for freq in frequencies])
        ref_counts = sum([freq[1] for freq in frequencies])
        # taking variant ids from the first client
        # they should be the same on each client
        variant_ids = frequencies[0][2]
        result = pandas.DataFrame(numpy.concatenate([alt_counts.reshape(-1, 1), ref_counts.reshape(-1, 1)], axis=1), 
                                  columns=['ALT_CTS', 'OBS_CT'], index=variant_ids)
        result.to_csv(self.freq_path, sep='\t')
        return ndarrays_to_parameters([alt_counts, ref_counts]), {}
    

class AlleleFreqClient(NumPyClient):
    """
    Client for computing allele frequencies
    """
    def __init__(self, pfile: str, aggregate_freq_file: str) -> None:
        self.pfile = pfile
        self.local_freq_file = self.pfile + '.acount'
        self.aggregate_freq_file = aggregate_freq_file
        
    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        
        run_plink(args_list=[
                    '--pfile', self.pfile,
                    '--freq', 'counts',
                    '--out', self.pfile
        ])
        
        frequencies = pandas.read_table(self.local_freq_file, index_col=0)
        variant_ids = frequencies['ID'].values.astype(numpy.dtype('U32'))
        return [frequencies['ALT_CTS'].values, frequencies['OBS_CT'].values, variant_ids], 1, {}
    
    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
        assert len(parameters) == 2
        alt_counts, ref_counts = parameters
        frequencies = pandas.read_table(self.local_freq_file, index_col=0)
        # #CHROM  ID      REF     ALT     ALT_CTS OBS_CT
        frequencies.loc[:, 'ALT_CTS'] = alt_counts
        frequencies.loc[:, 'OBS_CT'] = ref_counts
        frequencies.to_csv(self.aggregate_freq_file, sep='\t')
        print(f'Wrote aggregated frequencies from server to {self.aggregate_freq_file}')
        return 0.0, len(frequencies), {}
        

class FedPCAStrategy(FedAvg):
    def __init__(self, method: str, **kwargs) -> None:
        super().__init__(**kwargs)
        if method not in ['pstack', 'pcov']:
            raise ValueError(f'Unknown method {method}, should be one of "pstack", "pcov"')
        self.method = method
        
    def aggregate_fit(self, 
                      server_round: int, 
                      results: List[Tuple[ClientProxy, FitRes]], 
                      failures: List[Tuple[ClientProxy, FitRes]]) -> Tuple[Parameters, Dict[str, Scalar]]:
        
        if not self.accept_failures and failures:
            return None, {}
        
        if self.method == 'pstack':
            ndresults = [
                parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results
            ]
            components = [ndr[0] for ndr in ndresults]
            
            aggregated = numpy.concatenate(components, axis=0)
            
        elif self.method == 'pcov':
            components = [ndr[0] for ndr in ndresults]
            
            aggregated = sum(components)
            
        _, _, evectors = linalg.svds(aggregated, k=self.n_components)
        
        # Flip eigenvectors since 'linalg.svds' returns them in a reversed order
        evectors = numpy.flip(evectors, axis=0)
        return ndarrays_to_parameters([evectors])
        
        
class FedPCAServer:
    def __init__(self, args: ServerArgs):
        self.args = args
    
    def fit_transform(self, cache: FileCache):
        freq_path = cache.pfile_path().with_suffix('.acount')
        af_strategy = AlleleFreqStrategy(str(freq_path))
        config = ServerConfig(num_rounds=1)
        start_server(
                    server_address=f"[::]:{self.args.port}",
                    strategy=af_strategy,
                    config=config,
                    grpc_max_message_length=1536*1024*1024, # 1.5GB
        )
        
        pca_strategy = FedPCAStrategy(self.args.strategy)
        start_server(
                    server_address=f"[::]:{self.args.port}",
                    strategy=pca_strategy,
                    config=config,
                    grpc_max_message_length=1536*1024*1024, # 1.5GB
        )


@dataclass
class FedPCAClientArgs:
    method: str


class FedPCAClient(NumPyClient):
    def __init__(self, method: str, freq_file: str, pfile: str, output: str) -> None:
        super().__init__()
        self.method = method
        self.freq_file = freq_file
        self.pfile = pfile
        self.output = output

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        self.run_client_pca()
        if self.method == 'pstack':
            components, ids = self.load_pstack_component()
            print(f'Size of components is {components.nbytes // 1e+6}MB, size of ids is {ids.nbytes // 1e+6}MB')
            return [components], len(ids), {} 

        elif self.method == 'pcov':
            components = self.load_pcov_component()
            return [components], components.shape[0], {}
        
    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict[str, Scalar]]:
        raise RuntimeError('Not implemented')
        server_allele_file = os.path.join(
            self.result_folder, self.ALL, self.train_foldname_template % fold + '.eigenvec.allele'
        )

        for part in self.parts:
            sscore_file = self.cache
            sscore_file = os.path.join(self.result_folder, node, part % fold + '_projections.csv.eigenvec')

            run_plink(args_list=[
                '--pfile', self.pfile,
                '--read-freq', str(self.freq_file),
                '--score', server_allele_file, '2', '5',
                '--score-col-nums', f'6-{6 + self.n_components - 1}',
                '--out', sscore_file
            ])

        return super().evaluate(parameters, config)

    def run_client_pca(self):
        """
        Performs local PCA with plink
        """

        n_samples = len(pandas.read_csv(self.pfile + '.psam', sep='\t', header=0))
        run_plink(args_list=[
            '--pfile', self.pfile,
            '--read-freq', str(self.freq_file),
            '--pca', 'allele-wts', str(n_samples - 1),
            '--out', self.output,
        ])

    def load_pstack_component(self) -> Tuple[numpy.ndarray, numpy.ndarray]:
        evectors, evalues, ids = self.read_pca_results()
        evalues_matrix = numpy.zeros((len(evalues), len(evalues)))
        numpy.fill_diagonal(evalues_matrix, evalues)

        return numpy.dot(numpy.sqrt(evalues_matrix), evectors.T), ids

    def load_pcov_component(self):
        evectors, evalues, _ = self.read_pca_results()
        evalues_matrix = numpy.zeros((len(evalues), len(evalues)))
        numpy.fill_diagonal(evalues_matrix, evalues)

        return numpy.dot(numpy.dot(evectors, evalues_matrix), evectors.T)
    
    def read_pca_results(self):
        """
        Read plink results into NumPy arrays.
        """
        
        evalues = pandas.read_csv(self.output + '.eigenval', sep='\t', header=None)
        evectors = pandas.read_csv(self.output + '.eigenvec.allele', sep='\t', header=0)

        return evectors[evectors.columns[5:]].to_numpy(), evalues[0].to_numpy(), evectors['ID'].to_numpy(numpy.dtype('U32'))
    
    
class FedPCANode:
    def __init__(self, args: FedPCAClientArgs, node_args: NodeArgs) -> None:
        self.args = args
        self.node_args = node_args
        
    def fit_transform(self, cache: FileCache):
        
        self.allele_freq_client = AlleleFreqClient(str(cache.pfile_path(0, 'train')), 
                                                   str(cache.pfile_path(0, 'train').with_suffix('.aggregated')))
        
        print(f'Running allele frequency client on node')
        run_node(self.node_args, self.allele_freq_client)

        self.fed_pca_client = FedPCAClient(self.args.method, 
                                           str(cache.pfile_path(0, 'train').with_suffix('.aggregated')),
                                           str(cache.pfile_path(0, 'train')),
                                           str(cache.pfile_path(0, 'train')))

        print(f'Running federated PCA on node')
        run_node(self.node_args, self.fed_pca_client)
        

class FederatedPCASim:
    """
    Strategy:
    ---------

    1. Each node has serveral fold subsets: train, test, validation.
    2. Federated PCA is computed using only train subset of each node.
    3. Result projection is applied to all three subsets: train, test, validation.
    """

    # Results of federated PCA aggegation are stored for the <node identifier = ALL> on the filesystem
    ALL = 'ALL'

    def __init__(
        self,
        node_cache: FileCache,
        n_components=10,
        method='P-STACK',
        folds_num=5,
    ):
        """
        Two methods are available: P-COV and P-STACK. Both of them provide the same results but for the case
        when number of features >> number of samples, P-STACK's approach is more memory-efficient..
        """

        self.n_components = n_components
        self.method = method
        self.folds_num = folds_num

    def compute_allele_frequencies(self):
        """
        Compute centralized allele frequencies by joining plink *.acount files
        obtained separately for each node.
        """

        for node in self.nodes:
            for fold in range(self.folds_number):
                pfile = os.path.join(self.source_folder, node, self.train_foldname_template % fold)
                output = os.path.join(self.result_folder, node, self.train_foldname_template % fold)

                run_plink(args_list=[
                    '--pfile', pfile,
                    '--extract', self.variant_ids_file,
                    '--freq', 'counts',
                    '--out', output
                ])

        # Join allele frequency files for each fold
        for fold in range(self.folds_number):
            acount_data_list = []
            for node in self.nodes:
                acount_file = os.path.join(
                    self.result_folder, node, self.train_foldname_template % fold + '.acount'
                )

                acount_data_list.append(pd.read_csv(acount_file, sep='\t', header=0))

            result = acount_data_list[0]
            for acount_data in acount_data_list[1:]:
                # IDs consistency check before merge
                if not np.all(result['ID'] == acount_data['ID']):
                    raise ValueError('Variant IDs are not consistent between *.acount plink files')

                result['ALT_CTS'] += acount_data['ALT_CTS']
                result['OBS_CT'] += acount_data['OBS_CT']

            output_file = os.path.join(
                self.result_folder, self.ALL, self.train_foldname_template % fold + '.acount'
            )

            result.to_csv(output_file, sep='\t', index=False, header=True)

    def run(self):
        self.compute_allele_frequencies()

        for fold in range(self.folds_number):
            for node in self.nodes:
                self.run_client_pca(node, fold)

            # Aggregate client results
            self.run_server_aggregation(fold)

            for node in self.nodes + ['ALL']:
                self.run_client_projection(node, fold)

    def run_client_pca(self, node, fold):
        """
        Performs local PCA with plink
        """

        client_pfile = os.path.join(self.source_folder, node, self.train_foldname_template % fold)
        output_pca_file = os.path.join(self.result_folder, node, self.train_foldname_template % fold)
        allele_frequencies_file = os.path.join(
            self.result_folder, self.ALL, self.train_foldname_template % fold + '.acount'
        )

        n_samples = len(pd.read_csv(client_pfile + '.psam', sep='\t', header=0))
        run_plink(args_list=[
            '--pfile', client_pfile,
            '--extract', self.variant_ids_file,
            '--read-freq', allele_frequencies_file,
            '--pca', 'allele-wts', str(n_samples - 1),
            '--out', output_pca_file,
        ])

    def run_server_aggregation(self, fold):
        if self.method == 'P-STACK':
            aggregate, ids = self.load_pstack_component(self.nodes[0], fold)
            for node in self.nodes[1:]:
                component, other_ids = self.load_pstack_component(node, fold)

                # IDs consistency check before merge
                if not np.all(ids == other_ids):
                    raise ValueError('Variant IDs are not consistent between *.eigenvec.allele plink files')

                aggregate = np.concatenate([aggregate, component], axis=0)

        elif self.method == 'P-COV':
            aggregate = self.load_pcov_component(self.nodes[0], fold)
            for node in self.nodes[1:]:
                component = self.load_pcov_component(node, fold)
                aggregate = aggregate + component

        del component
        gc.collect()

        _, _, evectors = linalg.svds(aggregate, k=self.n_components)

        del aggregate
        gc.collect()

        # Flip eigenvectors since 'linalg.svds' returns them in a reversed order
        evectors = np.flip(evectors, axis=0)

        self.create_plink_eigenvec_allele_file(fold, evectors)

    def load_pstack_component(self, node, fold):
        evectors, evalues, ids = self.read_pca_results(node, fold)
        evalues_matrix = np.zeros((len(evalues), len(evalues)))
        np.fill_diagonal(evalues_matrix, evalues)

        return np.dot(np.sqrt(evalues_matrix), evectors.T), ids

    def load_pcov_component(self, node, fold):
        evectors, evalues, _ = self.read_pca_results(node, fold)
        evalues_matrix = np.zeros((len(evalues), len(evalues)))
        np.fill_diagonal(evalues_matrix, evalues)

        return np.dot(np.dot(evectors, evalues_matrix), evectors.T)

    def read_pca_results(self, node, fold):
        """
        Read plink results into NumPy arrays.
        """

        evalues_file = os.path.join(
            self.result_folder, node, self.train_foldname_template % fold + '.eigenval'
        )

        evectors_file = os.path.join(
            self.result_folder, node, self.train_foldname_template % fold + '.eigenvec.allele'
        )

        evalues = pd.read_csv(evalues_file, sep='\t', header=None)
        evectors = pd.read_csv(evectors_file, sep='\t', header=0)

        return evectors[evectors.columns[5:]].to_numpy(), evalues[0].to_numpy(), evectors['ID'].to_numpy()

    def create_plink_eigenvec_allele_file(self, fold, evectors):
        # Take the first 5 columns from one of the nodes to mimic combined plink .eigenvec.allele file
        client_allele_file = os.path.join(
            self.result_folder, self.nodes[0], self.train_foldname_template % fold + '.eigenvec.allele'
        )

        client_allele = pd.read_csv(client_allele_file, sep='\t', header=0)
        server_allele = client_allele[client_allele.columns[0:5]]

        for n in range(self.n_components):
            '''
            FIXME: Temporal solution

            Motivation
            ----------

            Principal components obtained from `svds` are normalized, i.e. have the norm the equals 1.
            Due to the normalization, resulting projection have small values for sample coordinates
            in the principal components space. Our classifier cannot learn from such values, since
            they are too small and we do not have a data normalization step.

            Normalization
            -------------

            Plink principal components are normalized in a different way. We use one of the client files
            to compute principal components norm and then renormalize the federated principal components
            in the same way to mimic plink PCA scale.

            This should be considered as a temporal solution. In the future data normalization may be
            added into the data loading stage, it can be implemented for our federated case as well.
            '''

            component_name = f'PC{n + 1}'
            current_norm = np.linalg.norm(evectors[n])
            plink_norm = np.linalg.norm(client_allele.iloc[:, n + 5])
            server_allele[component_name] = evectors[n] * plink_norm / current_norm

        server_allele_file = os.path.join(
            self.result_folder, self.ALL, self.train_foldname_template % fold + '.eigenvec.allele'
        )

        server_allele.to_csv(server_allele_file, sep='\t', header=True, index=False)


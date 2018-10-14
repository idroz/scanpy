from .. utils import doc_params
from ..tools._utils import choose_representation, doc_use_rep, doc_n_pcs
from .. import settings
from .. import logging as logg

@doc_params(doc_n_pcs=doc_n_pcs, use_rep=doc_use_rep)
def ivis(
    adata,
    use_rep=None,
    n_pcs=None,
    embedding_dims=2,
    k = 150,
    distance = 'pn',
    batch_size = 128,
    epochs = 1000,
    n_epochs_without_progress = 50,
    margin = 1, 
    ntrees = 50,
    search_k = -1,
    precompute = True,
    copy=False):
    """\
    ivis

    Parameters
    ----------
    adata : :class:`~anndata.AnnData`
        Annotated data matrix.
    {doc_n_pcs}
    {use_rep}
    embedding_dims : int, optional (default: 2)
        Number of dimensions in the embedding space
    
    k : int, optional (default: 150)
        The number of neighbours to retrieve for each point. Must be less than one minus the number of rows in the dataset.
    distance : string, optional (default: "pn")
        The loss function used to train the neural network. One of "pn", "euclidean", "softmax_ratio_pn", "softmax_ratio".
    
    batch_size : int, optional (default: 128)
        The size of mini-batches used during gradient descent while training the neural network. Must be less than the num_rows in the dataset.
    epochs : int, optional (default: 1000)
        The maximum number of epochs to train the model for. Each epoch the network will see a triplet based on each data-point once.
    n_epochs_without_progress : int, optional (default: 50)
        After n number of epochs without an improvement to the loss, terminate training early.
    margin : float, optional (default: 1)
        The distance that is enforced between points by the triplet loss functions
    ntrees : int, optional (default: 50)
        The number of random projections trees built by Annoy to approximate KNN. The more trees the higher the memory usage, but the better the accuracy of results.
    search_k : int, optional (default: -1)
        The maximum number of nodes inspected during a nearest neighbour query by Annoy. The higher, the more computation time required, but the higher the accuracy. The default 
        is n_trees * k, where k is the number of neighbours to retrieve. If this is set too low, a variable number of neighbours may be retrieved per data-point.
    precompute : boolean, optional (default: True)
        Whether to pre-compute the nearest neighbours. Pre-computing is significantly faster, but requires more memory. If memory is limited, try setting this to False.
    
    Returns
    -------
    Depending on `copy`, returns or updates `adata` with the following fields.

    X_ivis : `np.ndarray` (`adata.obs`, dtype `float`)
        IVIS coordinates of data.    
    """
    logg.info('computing IVIS', r=True)
    adata = adata.copy() if copy else adata

    X = choose_representation(adata, use_rep=use_rep, n_pcs=n_pcs)
    params_ivis = {'embedding_dims': embedding_dims,
                   'k': k,
                   'distance': distance,
                   'batch_size': batch_size,
                   'epochs': epochs,
                   'n_epochs_without_progress': n_epochs_without_progress,
                   'margin': margin,
                   'ntrees': ntrees,
                   'search_k': search_k,
                   'precompute': precompute
                   }
    from ivis import Ivis

    ivis_model = Ivis(**params_ivis)
    X_ivis = ivis_model.fit_transform(X)

    adata.obsm['X_ivis'] = X_ivis
    logg.info('    finished', time=True, end=' ' if settings.verbosity > 2 else '\n')
    logg.hint('added\n'
              '    \'X_ivis\', IVIS coordinates (adata.obsm)')
    return adata if copy else None
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
    X_ivis = ivis.fit_transform(X)

    adata.obsm['X_ivis'] = X_ivis
    logg.info('    finished', time=True, end=' ' if settings.verbosity > 2 else '\n')
    logg.hint('added\n'
              '    \'X_ivis\', IVIS coordinates (adata.obsm)')
    return adata if copy else None
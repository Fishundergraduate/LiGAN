import sys, os, pytest
from numpy import isclose, isnan
from torch import optim

sys.path.insert(0, '.')
import liGAN


@pytest.fixture(params=['AE', 'CE', 'VAE', 'CVAE'])
def solver(request):
    return getattr(
        liGAN.training, request.param + 'Solver'
    )(
        data_root='data/molport',
        train_file='data/molportFULL_rand_test0_1000.types',
        test_file='data/molportFULL_rand_test0_1000.types',
        batch_size=1000,
        rec_map_file='data/my_rec_map',
        lig_map_file='data/my_lig_map',
        resolution=1.0,
        grid_size=8,
        shuffle=False,
        random_rotation=False,
        random_translation=0,
        rec_molcache=None,
        lig_molcache=None,
        n_filters=5,
        width_factor=2,
        n_levels=3,
        conv_per_level=1,
        kernel_size=3,
        relu_leak=0.1,
        pool_type='a',
        unpool_type='n',
        pool_factor=2,
        n_latent=128,
        init_conv_pool=False,
        loss_weights=None,
        optim_type=optim.Adam,
        optim_kws=dict(
            lr=1e-5,
            betas=(0.9, 0.999),
        ),
        save_prefix='TEST',
        device='cuda'
    )


class TestSolver(object):

    def test_solver_init(self, solver):
        assert solver.curr_iter == 0
        for params in solver.parameters():
            assert params.detach().norm().cpu() > 0, 'params are zero'

    def test_solver_forward(self, solver):
        predictions, loss, metrics = solver.forward(solver.train_data)
        assert predictions.detach().norm().cpu() > 0, 'predictions are zero'
        assert not isclose(0, loss.item()), 'loss is zero'
        assert not isnan(loss.item()), 'loss is nan'

    def test_solver_step(self, solver):
        metrics0 = solver.step()
        _, _, metrics1 = solver.forward(solver.train_data)
        assert metrics1['loss'] < metrics0['loss'], 'loss did not decrease'

    def test_solver_test(self, solver):
        solver.test(1)
        assert solver.curr_iter == 0
        assert len(solver.metrics) == 1

    def test_solver_train(self, solver):
        solver.train(
            max_iter=10,
            test_interval=10,
            n_test_batches=1,
            save_interval=10,
        )
        assert solver.curr_iter == 10
        assert len(solver.metrics) == (1 + 10 + 1 + 1)
        loss_i = solver.metrics.loc[( 0, 'test'), 'loss'].mean()
        loss_f = solver.metrics.loc[(10, 'test'), 'loss'].mean()
        assert loss_f < loss_i, 'loss did not decrease'

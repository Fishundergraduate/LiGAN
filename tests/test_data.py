import sys, os, pytest
from numpy import isclose
from caffe.proto import caffe_pb2
os.environ['GLOG_minloglevel'] = '1'

sys.path.insert(0, '.')
from liGAN.data import AtomGridData
from liGAN.atom_types import AtomTyper


class TestAtomGridData(object):

    @pytest.fixture
    def data(self):
        return AtomGridData(
            data_root='data/molport',
            batch_size=10,
            rec_typer=AtomTyper.get_typer(prop_funcs='oad', radius_func='v'),
            lig_typer=AtomTyper.get_typer(prop_funcs='oad', radius_func='v'),
            resolution=0.5,
            dimension=23.5,
            shuffle=False,
        )

    @pytest.fixture
    def param(self):
        param = caffe_pb2.MolGridDataParameter()
        param.root_folder = 'data/molport'
        param.batch_size = 10
        param.recmap = 'data/my_rec_map'
        param.ligmap = 'data/my_lig_map'
        param.resolution = 0.5
        param.dimension = 23.5
        return param

    def test_data_init(self, data):
        assert data.n_rec_channels == 14
        assert data.n_lig_channels == 14
        assert data.ex_provider
        assert data.grid_maker
        assert data.grids.shape == (10, 14+14, 48, 48, 48)
        assert isclose(0, data.grids.norm().cpu())
        assert len(data) == 0

    def test_data_from_param(self, param):
        data = AtomGridData.from_param(param)
        assert data.n_rec_channels == 16
        assert data.n_lig_channels == 19
        assert data.ex_provider
        assert data.grid_maker
        assert data.grids.shape == (10, 16+19, 48, 48, 48)
        assert isclose(0, data.grids.norm().cpu())
        assert len(data) == 0

    def test_data_populate(self, data):
        data.populate('data/molportFULL_rand_test0_1000.types')
        assert len(data) == 1000

    def test_data_populate2(self, data):
        data.populate('data/molportFULL_rand_test0_1000.types')
        data.populate('data/molportFULL_rand_test0_1000.types')
        assert len(data) == 2000

    def test_data_forward_empty(self, data):
        with pytest.raises(AssertionError):
            data.forward()

    def test_data_forward_ok(self, data):
        data.populate('data/molportFULL_rand_test0_1000.types')
        grids, structs, labels = data.forward()
        rec_structs, lig_structs = structs
        assert grids.shape == (10, data.n_channels, 48, 48, 48)
        assert len(lig_structs) == 10
        assert labels.shape == (10,)
        assert not isclose(0, grids.norm().cpu())
        assert all(labels == 1)

    def test_data_forward_ligs(self, data):
        data.populate('data/molportFULL_rand_test0_1000.types')
        lig_grids, lig_structs, labels = data.forward(ligand_only=True)
        assert lig_grids.shape == (10, data.n_lig_channels, 48, 48, 48)
        assert len(lig_structs) == 10
        assert labels.shape == (10,)
        assert not isclose(0, lig_grids.norm().cpu())
        assert all(labels == 1)

    def test_data_forward_split(self, data):
        data.populate('data/molportFULL_rand_test0_1000.types')
        grids, structs, labels = data.forward(
            split_rec_lig=True
        )
        rec_grids, lig_grids = grids
        rec_structs, lig_structs = structs
        assert rec_grids.shape == (10, data.n_lig_channels, 48, 48, 48)
        assert lig_grids.shape == (10, data.n_rec_channels, 48, 48, 48)
        assert len(lig_structs) == 10
        assert labels.shape == (10,)
        assert not isclose(0, rec_grids.norm().cpu())
        assert not isclose(0, lig_grids.norm().cpu())
        assert all(labels == 1)

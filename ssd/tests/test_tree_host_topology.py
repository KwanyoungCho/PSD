import unittest
import numpy as np
from ssd.engine.helpers.tree_host_topology import context_topology


class TestTreeHostTopology(unittest.TestCase):
    def test_each_context_sees_exactly_its_path(self):
        parents = [-1,-1,0,1,2,1]
        depth,anc,rows = context_topology(parents)
        self.assertEqual(depth,[0,0,1,1,2,1])
        self.assertEqual(anc[4],[0,2])
        for j in range(len(parents)):
            path={0,j+1}
            p=parents[j]
            while p>=0:
                path.add(p+1)
                p=parents[p]
            self.assertEqual(set(np.flatnonzero(rows[j+1])),path)
        with self.assertRaises(ValueError):context_topology([-1,2])


if __name__=='__main__':unittest.main()

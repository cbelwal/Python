import unittest

import torch

from Algorithms.Alg_2_GenerateUserEmbeddings import (
    Algorithm_2_GenerateUserEmbeddings,
)
from Algorithms.TestData.CTestData_Simple import CTestData_Simple


class TestAlgorithm2Autoencoder(unittest.TestCase):
    def test_generates_shared_embeddings_with_expected_behavior(self):
        testData = CTestData_Simple()

        embeddings, losses = Algorithm_2_GenerateUserEmbeddings(
            embeddingDimensions=2,
            testData=testData,
        )

        self.assertEqual(tuple(embeddings.shape), (4, 2))
        self.assertEqual(tuple(losses.shape), (4,))
        self.assertTrue(torch.isfinite(embeddings).all())
        self.assertTrue(torch.isfinite(losses).all())
        self.assertTrue(torch.equal(embeddings[0], embeddings[2]))
        self.assertFalse(torch.equal(embeddings[0], embeddings[1]))

    def test_training_is_reproducible(self):
        testData = CTestData_Simple()

        firstEmbeddings, firstLosses = Algorithm_2_GenerateUserEmbeddings(
            embeddingDimensions=2,
            testData=testData,
        )
        secondEmbeddings, secondLosses = Algorithm_2_GenerateUserEmbeddings(
            embeddingDimensions=2,
            testData=testData,
        )

        self.assertTrue(torch.equal(firstEmbeddings, secondEmbeddings))
        self.assertTrue(torch.equal(firstLosses, secondLosses))


if __name__ == "__main__":
    unittest.main()

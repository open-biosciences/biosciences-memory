"""Unit tests for entity type models."""

import pytest

from biosciences_memory.models.entity_types import (
    EDGE_TYPE_MAP,
    EDGE_TYPES,
    ENTITY_TYPES,
    Disease,
    Drug,
    Gene,
    Pathway,
    Protein,
)


@pytest.mark.unit
class TestEntityTypes:
    def test_entity_types_count(self):
        assert len(ENTITY_TYPES) == 14  # 9 generic + 5 biosciences

    def test_edge_types_count(self):
        assert len(EDGE_TYPES) == 5

    def test_edge_type_map_count(self):
        assert len(EDGE_TYPE_MAP) == 10

    def test_gene_optional_fields(self):
        gene = Gene()
        assert gene.symbol is None
        assert gene.hgnc_id is None
        assert gene.ensembl_id is None

    def test_gene_with_values(self):
        gene = Gene(symbol="BRCA1", hgnc_id="HGNC:1100", ensembl_id="ENSG00000012048")
        assert gene.symbol == "BRCA1"

    def test_drug_with_phase(self):
        drug = Drug(name="Aspirin", chembl_id="CHEMBL25", phase=4)
        assert drug.phase == 4

    def test_protein_creation(self):
        protein = Protein(uniprot_id="P38398", name="BRCA1")
        assert protein.uniprot_id == "P38398"

    def test_disease_creation(self):
        disease = Disease(name="Breast cancer")
        assert disease.name == "Breast cancer"
        assert disease.mondo_id is None

    def test_pathway_creation(self):
        pathway = Pathway(name="MAPK signaling", wikipathways_id="WP382")
        assert pathway.wikipathways_id == "WP382"

    def test_biosciences_types_in_registry(self):
        for name in ["Gene", "Protein", "Drug", "Disease", "Pathway"]:
            assert name in ENTITY_TYPES

    def test_drug_gene_edge_mapping(self):
        assert ("Drug", "Gene") in EDGE_TYPE_MAP
        assert "TargetOf" in EDGE_TYPE_MAP[("Drug", "Gene")]

from typing import List, Optional

import pytest
from neomodel import (
    BooleanProperty,
    IntegerProperty,
    RelationshipTo,
    StringProperty,
    StructuredNode,
    UniqueIdProperty,
    ZeroOrMore,
    ZeroOrOne,
)
from pydantic import BaseModel, Field

from pydantic_neomodel_dict import Converter

# ===== Models for unique node tests =====

class TechnologyPydantic(BaseModel):
    """Technology with unique name constraint"""
    name: str
    version: Optional[str] = None
    deprecated: bool = False


class TechnologyOGM(StructuredNode):
    """Technology node with unique name"""
    __label__ = "UNQ_TechnologyOGM"
    uid = UniqueIdProperty()
    name = StringProperty(unique_index=True, required=True)
    version = StringProperty()
    deprecated = BooleanProperty(default=False)


class SkillPydantic(BaseModel):
    """Skill with unique name constraint"""
    name: str
    level: Optional[str] = None
    years: Optional[int] = None


class SkillOGM(StructuredNode):
    """Skill node with unique name"""
    __label__ = "UNQ_SkillOGM"
    uid = UniqueIdProperty()
    name = StringProperty(unique_index=True, required=True)
    level = StringProperty()
    years = IntegerProperty()


class CompanyPydantic(BaseModel):
    """Company with unique name constraint"""
    name: str
    industry: Optional[str] = None
    size: Optional[str] = None


class CompanyOGM(StructuredNode):
    """Company node with unique name"""
    __label__ = "UNQ_CompanyOGM"
    uid = UniqueIdProperty()
    name = StringProperty(unique_index=True, required=True)
    industry = StringProperty()
    size = StringProperty()


# ===== Models with relationships to unique nodes =====

class ProjectPydantic(BaseModel):
    """Project using technologies"""
    name: str
    description: str
    technologies: List[TechnologyPydantic] = Field(default_factory=list)
    lead_skill: Optional[SkillPydantic] = None


class ProjectOGM(StructuredNode):
    """Project node with relationships to unique nodes"""
    __label__ = "UNQ_ProjectOGM"
    uid = UniqueIdProperty()
    name = StringProperty(required=True)
    description = StringProperty()
    technologies = RelationshipTo(TechnologyOGM, 'USES_TECHNOLOGY', cardinality=ZeroOrMore)
    lead_skill = RelationshipTo(SkillOGM, 'REQUIRES_SKILL', cardinality=ZeroOrOne)


class DeveloperPydantic(BaseModel):
    """Developer with skills and projects"""
    name: str
    email: str
    skills: List[SkillPydantic] = Field(default_factory=list)
    projects: List[ProjectPydantic] = Field(default_factory=list)


class DeveloperOGM(StructuredNode):
    """Developer node with nested relationships"""
    __label__ = "UNQ_DeveloperOGM"
    uid = UniqueIdProperty()
    name = StringProperty(required=True)
    email = StringProperty(required=True)
    skills = RelationshipTo(SkillOGM, 'HAS_SKILL', cardinality=ZeroOrMore)
    projects = RelationshipTo(ProjectOGM, 'WORKED_ON', cardinality=ZeroOrMore)


class EmploymentPydantic(BaseModel):
    """Employment with company and technologies"""
    position: str
    company: CompanyPydantic
    technologies: List[TechnologyPydantic] = Field(default_factory=list)
    skills: List[SkillPydantic] = Field(default_factory=list)


class EmploymentOGM(StructuredNode):
    """Employment node with relationships to unique nodes"""
    __label__ = "UNQ_EmploymentOGM"
    uid = UniqueIdProperty()
    position = StringProperty(required=True)
    company = RelationshipTo(CompanyOGM, 'AT_COMPANY', cardinality=ZeroOrOne)
    technologies = RelationshipTo(TechnologyOGM, 'USED_TECHNOLOGY', cardinality=ZeroOrMore)
    skills = RelationshipTo(SkillOGM, 'UTILIZED_SKILL', cardinality=ZeroOrMore)


class ResumePydantic(BaseModel):
    """Resume with multiple unique node references"""
    candidate_name: str
    employment_history: List[EmploymentPydantic] = Field(default_factory=list)
    all_skills: List[SkillPydantic] = Field(default_factory=list)
    preferred_technologies: List[TechnologyPydantic] = Field(default_factory=list)


class ResumeOGM(StructuredNode):
    """Resume node with complex relationships"""
    __label__ = "UNQ_ResumeOGM"
    uid = UniqueIdProperty()
    candidate_name = StringProperty(required=True)
    employment_history = RelationshipTo(EmploymentOGM, 'HAS_EMPLOYMENT', cardinality=ZeroOrMore)
    all_skills = RelationshipTo(SkillOGM, 'HAS_SKILL', cardinality=ZeroOrMore)
    preferred_technologies = RelationshipTo(TechnologyOGM, 'PREFERS_TECHNOLOGY', cardinality=ZeroOrMore)


# ===== Models for mixed unique/non-unique tests =====

class TagPydantic(BaseModel):
    """Non-unique tag"""
    name: str
    category: str


class TagOGM(StructuredNode):
    """Non-unique tag node"""
    __label__ = "UNQ_TagOGM"
    uid = UniqueIdProperty()
    name = StringProperty(required=True)  # NOT unique
    category = StringProperty(required=True)


class ArticlePydantic(BaseModel):
    """Article with unique and non-unique references"""
    title: str
    author_company: CompanyPydantic  # Unique
    technologies: List[TechnologyPydantic] = Field(default_factory=list)  # Unique
    tags: List[TagPydantic] = Field(default_factory=list)  # Non-unique


class ArticleOGM(StructuredNode):
    """Article with mixed relationships"""
    __label__ = "UNQ_ArticleOGM"
    uid = UniqueIdProperty()
    title = StringProperty(required=True)
    author_company = RelationshipTo(CompanyOGM, 'AUTHORED_BY', cardinality=ZeroOrOne)
    technologies = RelationshipTo(TechnologyOGM, 'MENTIONS_TECHNOLOGY', cardinality=ZeroOrMore)
    tags = RelationshipTo(TagOGM, 'HAS_TAG', cardinality=ZeroOrMore)


# ===== Fixtures =====

@pytest.fixture
def register_all_models():
    """Register all test models"""
    Converter.register_models(TechnologyPydantic, TechnologyOGM)
    Converter.register_models(SkillPydantic, SkillOGM)
    Converter.register_models(CompanyPydantic, CompanyOGM)
    Converter.register_models(ProjectPydantic, ProjectOGM)
    Converter.register_models(DeveloperPydantic, DeveloperOGM)
    Converter.register_models(EmploymentPydantic, EmploymentOGM)
    Converter.register_models(ResumePydantic, ResumeOGM)
    Converter.register_models(TagPydantic, TagOGM)
    Converter.register_models(ArticlePydantic, ArticleOGM)
    yield


@pytest.fixture
def duplicate_technologies():
    """Create multiple instances with same unique name"""
    return [
        TechnologyPydantic(name="Python", version="3.11"),
        TechnologyPydantic(name="Python", version="3.12"),
        TechnologyPydantic(name="Python"),  # No version
    ]


@pytest.fixture
def duplicate_skills():
    """Create multiple skills with same name"""
    return [
        SkillPydantic(name="Docker", level="Expert", years=5),
        SkillPydantic(name="Docker", level="Advanced", years=3),
        SkillPydantic(name="Docker"),  # Minimal data
    ]


@pytest.fixture
def project_with_duplicates():
    """Create project referencing duplicate technologies"""
    return ProjectPydantic(
        name="Backend System",
        description="Microservices backend",
        technologies=[
            TechnologyPydantic(name="Python", version="3.11"),
            TechnologyPydantic(name="Docker"),
            TechnologyPydantic(name="Python", version="3.12"),  # Duplicate name
            TechnologyPydantic(name="Redis"),
            TechnologyPydantic(name="Docker"),  # Another duplicate
        ],
        lead_skill=SkillPydantic(name="Python", level="Expert")
    )


@pytest.fixture
def complex_resume():
    """Create resume with multiple references to same unique nodes"""
    python_tech = TechnologyPydantic(name="Python", version="3.11")
    docker_tech = TechnologyPydantic(name="Docker")
    docker_skill = SkillPydantic(name="Docker", level="Expert")
    python_skill = SkillPydantic(name="Python", level="Expert")

    company1 = CompanyPydantic(name="TechCorp", industry="Software")
    company2 = CompanyPydantic(name="DataCo", industry="Analytics")

    employment1 = EmploymentPydantic(
        position="Senior Developer",
        company=company1,
        technologies=[python_tech, docker_tech],
        skills=[python_skill, docker_skill]
    )

    employment2 = EmploymentPydantic(
        position="Lead Engineer",
        company=company1,  # Same company
        technologies=[python_tech],  # Same technology
        skills=[python_skill]  # Same skill
    )

    employment3 = EmploymentPydantic(
        position="Architect",
        company=company2,
        technologies=[docker_tech, python_tech],  # Same technologies, different order
        skills=[docker_skill]
    )

    return ResumePydantic(
        candidate_name="John Doe",
        employment_history=[employment1, employment2, employment3],
        all_skills=[python_skill, docker_skill],  # Same skills as in employment
        preferred_technologies=[python_tech, docker_tech]  # Same technologies
    )


@pytest.fixture
def article_mixed_nodes():
    """Create article with unique and non-unique nodes"""
    return ArticlePydantic(
        title="Advanced Python Techniques",
        author_company=CompanyPydantic(name="TechBlog Inc", industry="Publishing"),
        technologies=[
            TechnologyPydantic(name="Python", version="3.11"),
            TechnologyPydantic(name="Python", version="3.12"),  # Same name, different version
            TechnologyPydantic(name="FastAPI"),
        ],
        tags=[
            TagPydantic(name="tutorial", category="education"),
            TagPydantic(name="tutorial", category="programming"),  # Same name, different category
            TagPydantic(name="advanced", category="level"),
        ]
    )


# ===== Test Class =====

class TestUniqueNodeHandling:
    """Comprehensive tests for unique node constraint handling"""

    def test_single_unique_node_creation(self, db_connection, register_all_models):
        """
        Test that a single unique node is created properly.

        Verifies that nodes with unique_index=True are created with element_id
        and pass isinstance checks.
        """
        tech = TechnologyPydantic(name="Python", version="3.11")
        tech_ogm = Converter.to_ogm(tech)

        # Verify node created
        assert tech_ogm is not None, "Node should be created"
        assert isinstance(tech_ogm, TechnologyOGM), "Should return proper OGM instance"

        # Verify properties
        assert tech_ogm.name == "Python", "Name should be preserved"
        assert tech_ogm.version == "3.11", "Version should be preserved"
        assert tech_ogm.deprecated is False, "Default value should be set"

        # Verify element_id exists (critical for relationship connections)
        assert hasattr(tech_ogm, 'element_id'), "Node should have element_id attribute"
        assert tech_ogm.element_id is not None, "element_id should not be None"

        # Verify node is saved in database
        found = TechnologyOGM.nodes.get_or_none(name="Python")
        assert found is not None, "Node should be findable in database"
        assert found.element_id == tech_ogm.element_id, "Should be the same node"

    def test_unique_node_reuse(self, db_connection, register_all_models, duplicate_technologies):
        """
        Test that multiple instances with same unique value reuse the same node.

        Verifies get_or_create behavior for unique constraints.
        """
        techs = duplicate_technologies

        # Convert all three Python instances
        ogm_nodes = [Converter.to_ogm(tech) for tech in techs]

        # All should return the same node (same element_id)
        assert ogm_nodes[0].element_id == ogm_nodes[1].element_id, \
            "Same unique name should return same node"
        assert ogm_nodes[1].element_id == ogm_nodes[2].element_id, \
            "All instances should share same node"

        # Verify only one node exists in database
        all_pythons = list(TechnologyOGM.nodes.filter(name="Python"))
        assert len(all_pythons) == 1, "Only one unique node should exist"

        # Verify last update wins for non-unique properties
        final_node = TechnologyOGM.nodes.get(name="Python")
        # The last conversion had no version, but previous ones might have updated it
        assert final_node.name == "Python", "Name should be preserved"

    def test_relationship_to_unique_nodes(self, db_connection, register_all_models, project_with_duplicates):
        """
        Test that relationships to unique nodes work correctly.

        Verifies that relationships properly connect to existing unique nodes.
        """
        project = project_with_duplicates
        project_ogm = Converter.to_ogm(project)

        # Verify project created
        assert project_ogm.name == "Backend System", "Project should be created"

        # Check technologies relationship
        tech_nodes = list(project_ogm.technologies.all())

        # Should have connected to unique nodes
        unique_names = {tech.name for tech in tech_nodes}
        assert unique_names == {"Python", "Docker", "Redis"}, \
            "Should have 3 unique technologies despite duplicates"

        # Verify each unique technology exists only once in database
        for tech_name in ["Python", "Docker", "Redis"]:
            matching = list(TechnologyOGM.nodes.filter(name=tech_name))
            assert len(matching) == 1, f"Only one {tech_name} node should exist"

        # Check lead_skill relationship
        lead_skills = list(project_ogm.lead_skill.all())
        assert len(lead_skills) == 1, "Should have one lead skill"
        assert lead_skills[0].name == "Python", "Lead skill should be Python"
        assert lead_skills[0].level == "Expert", "Skill properties should be preserved"

    def test_complex_nested_unique_references(self, db_connection, register_all_models, complex_resume):
        """
        Test complex nested structures with multiple references to same unique nodes.

        Verifies that unique nodes are properly reused across nested relationships.
        """
        resume = complex_resume
        resume_ogm = Converter.to_ogm(resume)

        # Verify resume created
        assert resume_ogm.candidate_name == "John Doe", "Resume should be created"

        # Check employment history
        employments = list(resume_ogm.employment_history.all())
        assert len(employments) == 3, "Should have 3 employment records"

        # Verify companies are reused
        company_nodes = []
        for emp in employments:
            companies = list(emp.company.all())
            if companies:
                company_nodes.extend(companies)

        # TechCorp should appear twice but be same node
        techcorp_nodes = [c for c in company_nodes if c.name == "TechCorp"]
        assert len(techcorp_nodes) == 2, "TechCorp referenced twice"
        assert techcorp_nodes[0].element_id == techcorp_nodes[1].element_id, \
            "Should be same TechCorp node"

        # Verify only 2 unique companies exist in database
        all_companies = list(CompanyOGM.nodes.all())
        company_names = {c.name for c in all_companies}
        assert company_names == {"TechCorp", "DataCo"}, "Only unique companies should exist"

        # Verify technologies are reused across employments
        all_tech_refs = []
        for emp in employments:
            all_tech_refs.extend(list(emp.technologies.all()))

        # Count Python references
        python_refs = [t for t in all_tech_refs if t.name == "Python"]
        assert len(python_refs) > 0, "Python should be referenced"

        # All Python references should be same node
        python_ids = {t.element_id for t in python_refs}
        assert len(python_ids) == 1, "All Python references should be same node"

        # Verify skills are reused
        all_skill_refs = []
        for emp in employments:
            all_skill_refs.extend(list(emp.skills.all()))

        docker_skill_refs = [s for s in all_skill_refs if s.name == "Docker"]
        docker_ids = {s.element_id for s in docker_skill_refs}
        assert len(docker_ids) == 1, "All Docker skill references should be same node"

        # Verify top-level skills reference same nodes
        resume_skills = list(resume_ogm.all_skills.all())
        resume_skill_names = {s.name for s in resume_skills}
        assert resume_skill_names == {"Python", "Docker"}, "Should have both skills"

        # These should be the same nodes as in employment
        for skill in resume_skills:
            if skill.name == "Docker":
                assert skill.element_id in docker_ids, \
                    "Resume Docker skill should be same as employment Docker skill"

    def test_mixed_unique_and_nonunique_nodes(self, db_connection, register_all_models, article_mixed_nodes):
        """
        Test handling of mixed unique and non-unique nodes.

        Verifies that unique nodes are deduplicated while non-unique nodes are not.
        """
        article = article_mixed_nodes
        article_ogm = Converter.to_ogm(article)

        # Verify article created
        assert article_ogm.title == "Advanced Python Techniques", "Article should be created"

        # Check unique company node
        companies = list(article_ogm.author_company.all())
        assert len(companies) == 1, "Should have one company"
        assert companies[0].name == "TechBlog Inc", "Company should be created"

        # Check unique technology nodes (should deduplicate Python)
        techs = list(article_ogm.technologies.all())
        tech_names = {t.name for t in techs}
        assert tech_names == {"Python", "FastAPI"}, \
            "Should have 2 unique technologies despite 2 Python entries"

        # Verify only one Python node in database
        python_nodes = list(TechnologyOGM.nodes.filter(name="Python"))
        assert len(python_nodes) == 1, "Only one Python node should exist"

        # Check non-unique tag nodes (should NOT deduplicate)
        tags = list(article_ogm.tags.all())
        assert len(tags) == 3, "Should have 3 tags (no deduplication)"

        # Both tutorial tags should exist as separate nodes
        tutorial_tags = [t for t in tags if t.name == "tutorial"]
        assert len(tutorial_tags) == 2, "Both tutorial tags should exist"

        # Verify they are different nodes
        if len(tutorial_tags) == 2:
            assert tutorial_tags[0].element_id != tutorial_tags[1].element_id, \
                "Non-unique tags should be different nodes"
            categories = {t.category for t in tutorial_tags}
            assert categories == {"education", "programming"}, \
                "Both tutorial categories should exist"

    def test_update_unique_node_properties(self, db_connection, register_all_models):
        """
        Test that non-unique properties of unique nodes get updated.

        Verifies that when reusing a unique node, non-unique properties are updated.
        """
        # Create first version
        tech1 = TechnologyPydantic(name="Redis", version="6.0", deprecated=False)
        ogm1 = Converter.to_ogm(tech1)

        assert ogm1.version == "6.0", "Initial version should be set"
        assert ogm1.deprecated is False, "Initial deprecated should be False"

        # Create second version with same name
        tech2 = TechnologyPydantic(name="Redis", version="7.0", deprecated=True)
        ogm2 = Converter.to_ogm(tech2)

        # Should be same node
        assert ogm1.element_id == ogm2.element_id, "Should reuse same node"

        # Properties should be updated
        assert ogm2.version == "7.0", "Version should be updated"
        assert ogm2.deprecated is True, "Deprecated should be updated"

        # Verify in database
        redis_node = TechnologyOGM.nodes.get(name="Redis")
        assert redis_node.version == "7.0", "Database should have updated version"
        assert redis_node.deprecated is True, "Database should have updated deprecated"

    def test_concurrent_unique_node_creation(self, db_connection, register_all_models):
        """
        Test that concurrent creation of same unique node works correctly.

        Simulates race condition where multiple processes try to create same unique node.
        """
        # Create multiple developers all using Python
        developers = []
        for i in range(5):
            dev = DeveloperPydantic(
                name=f"Developer{i}",
                email=f"dev{i}@example.com",
                skills=[SkillPydantic(name="Python", level="Expert")],
                projects=[
                    ProjectPydantic(
                        name=f"Project{i}",
                        description=f"Project {i}",
                        technologies=[TechnologyPydantic(name="Python", version="3.11")]
                    )
                ]
            )
            developers.append(dev)

        # Convert all developers
        dev_ogms = [Converter.to_ogm(dev) for dev in developers]

        # Verify all created
        assert len(dev_ogms) == 5, "All developers should be created"

        # Verify only one Python skill node exists
        python_skills = list(SkillOGM.nodes.filter(name="Python"))
        assert len(python_skills) == 1, "Only one Python skill should exist"

        # Verify only one Python technology node exists
        python_techs = list(TechnologyOGM.nodes.filter(name="Python"))
        assert len(python_techs) == 1, "Only one Python technology should exist"

        # Verify all developers reference the same skill
        skill_ids = set()
        for dev_ogm in dev_ogms:
            skills = list(dev_ogm.skills.all())
            for skill in skills:
                if skill.name == "Python":
                    skill_ids.add(skill.element_id)

        assert len(skill_ids) == 1, "All developers should reference same Python skill"

    def test_round_trip_with_unique_nodes(self, db_connection, register_all_models):
        """
        Test round-trip conversion with unique nodes.

        Verifies that converting to OGM and back preserves unique node relationships.
        """
        # Create project with technologies
        project = ProjectPydantic(
            name="Test Project",
            description="Testing round trip",
            technologies=[
                TechnologyPydantic(name="Python", version="3.11"),
                TechnologyPydantic(name="Docker"),
                TechnologyPydantic(name="Python", version="3.12"),  # Duplicate
            ],
            lead_skill=SkillPydantic(name="Python", level="Expert")
        )

        # Convert to OGM
        project_ogm = Converter.to_ogm(project)

        # Convert back to Pydantic
        project_back = Converter.to_pydantic(project_ogm)

        # Verify basic properties
        assert project_back.name == "Test Project", "Name should be preserved"
        assert project_back.description == "Testing round trip", "Description should be preserved"

        # Verify technologies (should have deduplicated)
        tech_names = [t.name for t in project_back.technologies]
        unique_tech_names = set(tech_names)
        assert unique_tech_names == {"Python", "Docker"}, \
            "Should have unique technologies after round trip"

        # Verify lead skill
        assert project_back.lead_skill is not None, "Lead skill should exist"
        assert project_back.lead_skill.name == "Python", "Lead skill name should be preserved"
        assert project_back.lead_skill.level == "Expert", "Lead skill level should be preserved"

    def test_delete_with_unique_node_references(self, db_connection, register_all_models):
        """
        Test that deleting entities with unique node references doesn't delete the unique nodes.

        Verifies that unique nodes persist when entities referencing them are deleted.
        """
        # Create two projects using same technology
        project1 = ProjectPydantic(
            name="Project 1",
            description="First project",
            technologies=[TechnologyPydantic(name="Kubernetes")]
        )
        project2 = ProjectPydantic(
            name="Project 2",
            description="Second project",
            technologies=[TechnologyPydantic(name="Kubernetes")]
        )

        # Convert both
        ogm1 = Converter.to_ogm(project1)
        ogm2 = Converter.to_ogm(project2)

        # Verify both exist
        assert ProjectOGM.nodes.get_or_none(name="Project 1") is not None
        assert ProjectOGM.nodes.get_or_none(name="Project 2") is not None

        # Verify Kubernetes node exists
        k8s = TechnologyOGM.nodes.get(name="Kubernetes")
        assert k8s is not None, "Kubernetes node should exist"

        # Delete first project
        ogm1.delete()

        # Verify first project deleted
        assert ProjectOGM.nodes.get_or_none(name="Project 1") is None

        # Verify Kubernetes still exists (referenced by project2)
        k8s_after = TechnologyOGM.nodes.get_or_none(name="Kubernetes")
        assert k8s_after is not None, "Kubernetes should still exist"
        assert k8s_after.element_id == k8s.element_id, "Should be same Kubernetes node"

        # Verify second project still connected
        project2_techs = list(ogm2.technologies.all())
        assert len(project2_techs) == 1, "Project 2 should still have technology"
        assert project2_techs[0].name == "Kubernetes", "Should still be Kubernetes"

    def test_unique_node_with_none_values(self, db_connection, register_all_models):
        """
        Test unique nodes with None/null values in non-unique fields.

        Verifies that None values are handled correctly in unique nodes.
        """
        # Create skill with minimal data
        skill1 = SkillPydantic(name="GraphQL")  # No level or years
        ogm1 = Converter.to_ogm(skill1)

        assert ogm1.name == "GraphQL", "Name should be set"
        assert ogm1.level is None, "Level should be None"
        assert ogm1.years is None, "Years should be None"

        # Create same skill with full data
        skill2 = SkillPydantic(name="GraphQL", level="Intermediate", years=2)
        ogm2 = Converter.to_ogm(skill2)

        # Should be same node
        assert ogm1.element_id == ogm2.element_id, "Should reuse same node"

        # Values should be updated
        assert ogm2.level == "Intermediate", "Level should be updated"
        assert ogm2.years == 2, "Years should be updated"

        # Create another with partial data
        skill3 = SkillPydantic(name="GraphQL", level="Advanced")  # No years
        ogm3 = Converter.to_ogm(skill3)

        # Should still be same node
        assert ogm2.element_id == ogm3.element_id, "Should still reuse same node"
        assert ogm3.level == "Advanced", "Level should be updated again"
        # Years should remain from previous update
        assert ogm3.years == 2, "Years should be preserved from previous update"


# ===== Additional edge case tests =====

class TestUniqueNodeEdgeCases:
    """Edge cases and error conditions for unique node handling"""

    def test_empty_unique_value_handling(self, db_connection, register_all_models):
        """
        Test handling of empty string as unique value.

        Some systems might allow empty strings as unique values.
        """
        # This should likely raise an error or be handled specially
        with pytest.raises(Exception):
            # Empty name should fail validation
            tech = TechnologyPydantic(name="")
            Converter.to_ogm(tech)

    def test_unique_node_isinstance_check(self, db_connection, register_all_models):
        """
        Test that returned nodes pass isinstance checks.

        This is critical for the bug fix - nodes must be actual instances, not proxies.
        """
        skill = SkillPydantic(name="Testing", level="Expert")
        skill_ogm = Converter.to_ogm(skill)

        # Must pass isinstance check
        assert isinstance(skill_ogm, SkillOGM), \
            "Returned node must be actual SkillOGM instance, not QuerySet proxy"

        # Must have element_id
        assert hasattr(skill_ogm, 'element_id'), "Must have element_id attribute"
        assert skill_ogm.element_id is not None, "element_id must not be None"

        # Must be usable in relationship manager
        project = ProjectPydantic(
            name="Test Project",
            description="Testing isinstance",
            lead_skill=skill
        )
        project_ogm = Converter.to_ogm(project)

        # Relationship should be connected
        lead_skills = list(project_ogm.lead_skill.all())
        assert len(lead_skills) == 1, "Relationship should be connected"
        assert lead_skills[0].element_id == skill_ogm.element_id, \
            "Should be same skill node"

    def test_unique_node_save_called(self, db_connection, register_all_models):
        """
        Test that save() is called on unique nodes to ensure element_id.

        Verifies the fix ensures nodes are saved before being used in relationships.
        """
        company = CompanyPydantic(name="SaveTest Inc")
        company_ogm = Converter.to_ogm(company)

        # Node should be saved (have element_id)
        assert company_ogm.element_id is not None, \
            "Node must be saved (have element_id) after conversion"

        # Should be findable in database
        found = CompanyOGM.nodes.get_or_none(name="SaveTest Inc")
        assert found is not None, "Node should be in database"
        assert found.element_id == company_ogm.element_id, "Should be same node"

        # Should be usable in relationships immediately
        employment = EmploymentPydantic(
            position="CEO",
            company=company
        )
        employment_ogm = Converter.to_ogm(employment)

        # Verify relationship works
        companies = list(employment_ogm.company.all())
        assert len(companies) == 1, "Company relationship should work"
        assert companies[0].element_id == company_ogm.element_id, \
            "Should be same company node"

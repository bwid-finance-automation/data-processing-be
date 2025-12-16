"""Base repository implementation with generic CRUD operations."""

from typing import Generic, TypeVar, Optional, List, Type, Any
from sqlalchemy import select, update, delete, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.infrastructure.database.base import Base

# Generic type for SQLAlchemy models
ModelType = TypeVar("ModelType", bound=Base)


class BaseRepository(Generic[ModelType]):
    """
    Base repository class providing generic CRUD operations.

    Usage:
        class UserRepository(BaseRepository[UserModel]):
            def __init__(self, session: AsyncSession):
                super().__init__(UserModel, session)
    """

    def __init__(self, model: Type[ModelType], session: AsyncSession):
        """
        Initialize repository with model class and database session.

        Args:
            model: SQLAlchemy model class
            session: Async database session
        """
        self.model = model
        self.session = session

    async def get(self, id: int) -> Optional[ModelType]:
        """
        Get a single record by ID.

        Args:
            id: Primary key ID

        Returns:
            Model instance or None if not found
        """
        result = await self.session.execute(
            select(self.model).where(self.model.id == id)
        )
        return result.scalar_one_or_none()

    async def get_by_uuid(self, uuid: str) -> Optional[ModelType]:
        """
        Get a single record by UUID (if model has uuid column).

        Args:
            uuid: UUID string

        Returns:
            Model instance or None if not found
        """
        if not hasattr(self.model, 'uuid'):
            raise AttributeError(f"{self.model.__name__} does not have a uuid column")

        result = await self.session.execute(
            select(self.model).where(self.model.uuid == uuid)
        )
        return result.scalar_one_or_none()

    async def get_all(
        self,
        skip: int = 0,
        limit: int = 100,
        order_by: Optional[Any] = None,
    ) -> List[ModelType]:
        """
        Get all records with pagination.

        Args:
            skip: Number of records to skip
            limit: Maximum number of records to return
            order_by: Column to order by (default: id desc)

        Returns:
            List of model instances
        """
        query = select(self.model)

        if order_by is not None:
            query = query.order_by(order_by)
        else:
            query = query.order_by(self.model.id.desc())

        query = query.offset(skip).limit(limit)

        result = await self.session.execute(query)
        return list(result.scalars().all())

    async def create(self, obj: ModelType) -> ModelType:
        """
        Create a new record.

        Args:
            obj: Model instance to create

        Returns:
            Created model instance with ID populated
        """
        self.session.add(obj)
        await self.session.flush()
        await self.session.refresh(obj)
        return obj

    async def create_many(self, objects: List[ModelType]) -> List[ModelType]:
        """
        Create multiple records.

        Args:
            objects: List of model instances to create

        Returns:
            List of created model instances
        """
        self.session.add_all(objects)
        await self.session.flush()
        for obj in objects:
            await self.session.refresh(obj)
        return objects

    async def update(self, id: int, **kwargs) -> Optional[ModelType]:
        """
        Update a record by ID.

        Args:
            id: Primary key ID
            **kwargs: Fields to update

        Returns:
            Updated model instance or None if not found
        """
        await self.session.execute(
            update(self.model)
            .where(self.model.id == id)
            .values(**kwargs)
        )
        await self.session.flush()
        return await self.get(id)

    async def delete(self, id: int) -> bool:
        """
        Delete a record by ID (hard delete).

        Args:
            id: Primary key ID

        Returns:
            True if deleted, False if not found
        """
        result = await self.session.execute(
            delete(self.model).where(self.model.id == id)
        )
        await self.session.flush()
        return result.rowcount > 0

    async def soft_delete(self, id: int) -> Optional[ModelType]:
        """
        Soft delete a record by ID (if model supports it).

        Args:
            id: Primary key ID

        Returns:
            Updated model instance or None if not found
        """
        if not hasattr(self.model, 'is_deleted'):
            raise AttributeError(f"{self.model.__name__} does not support soft delete")

        from datetime import datetime
        return await self.update(
            id,
            is_deleted=True,
            deleted_at=datetime.utcnow()
        )

    async def count(self) -> int:
        """
        Count total records.

        Returns:
            Total number of records
        """
        result = await self.session.execute(
            select(func.count()).select_from(self.model)
        )
        return result.scalar_one()

    async def exists(self, id: int) -> bool:
        """
        Check if a record exists by ID.

        Args:
            id: Primary key ID

        Returns:
            True if exists, False otherwise
        """
        result = await self.session.execute(
            select(func.count())
            .select_from(self.model)
            .where(self.model.id == id)
        )
        return result.scalar_one() > 0

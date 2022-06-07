# Requires mysql-connector-python
from mysql.connector import CMySQLConnection
import mysql.connector
from mysql.connector.cursor_cext import CMySQLCursor
import json


# Database interaction object
# Used just for simplifying MySQL queries
# Not needed outside of Place class
class DB:
	"""Database interaction object"""
	_db: CMySQLConnection = None
	_cursor: CMySQLCursor = None
	lastQuery: list = []

	def __init__(self,
	             host = "localhost",
	             port = 3306,
	             database = "me_project",
	             user = "root",
	             password = "password"
	             ):
		self._db = mysql.connector.connect(
			host = host,
			database = database,
			port = port,
			user = user,
			password = password
		)

		self._cursor = self._db.cursor()

	def query(self, query: str, change: bool = False) -> list:
		self._cursor.execute(query)  # send query
		if change:
			self._db.commit()  # used for applying changes if changing anything
		self.lastQuery = self._cursor.fetchall()
		return self.lastQuery

	def close(self) -> None:
		"""Close MySql connection"""
		self._db.close()
		del self._db
		del self.lastQuery
		del self._cursor


db = DB()


class Place:
	""" """
	id: int  # id not necessary since auto_increment
	sector: str
	name: str
	rating: float
	workingHours: str
	address: str
	specialization: str
	phone: str
	hasAlcohol: bool
	hasDelivery: bool
	hasPark: bool
	coordinates: str

	def __init__(self,
	             sector: str = "",
	             name: str = "",
	             rating: float = 0.0,
	             workingHours: str = "",
	             address: str = "",
	             specialization: str = "",
	             phone: str = "",
	             hasAlcohol: bool = False,
	             hasDelivery: bool = False,
	             hasPark: bool = False,
	             coordinates: str = ""
	             ):
		self.sector = sector
		self.name = name
		self.rating = rating
		self.workingHours = workingHours
		self.address = address
		self.specialization = specialization
		self.phone = phone
		self.hasAlcohol = hasAlcohol
		self.hasDelivery = hasDelivery
		self.hasPark = hasPark
		self.coordinates = coordinates
		db = DB()

	def toDict(self):
		d = dict()
		d["sector"] = str(self.sector)
		d["name"] = str(self.name)
		d["rating"] = str(self.rating)
		d["workingHours"] = str(self.workingHours)
		d["address"] = str(self.address)
		d["specialization"] = str(self.specialization)
		d["phone"] = str(self.phone)
		d["hasAlcohol"] = str(self.hasAlcohol)
		d["hasDelivery"] = str(self.hasDelivery)
		d["hasPark"] = str(self.hasPark)
		d["coordinates"] = str(self.coordinates)
		return d

	@staticmethod
	def to_JSON_list(places: list):
		ls = list()
		for p in places:
			ls.append(json.dumps(p.toDict()))
		return ls

	@staticmethod
	def fromList(ls: list):
		"""Construct place object from tuple"""
		if len(ls) == 1:  # if it is a list/tuple containing other list/tuple
			ls = ls[0]
		place = Place()
		place.id = ls[0]
		place.sector = ls[1]
		place.name = ls[2]
		place.rating = ls[3]
		place.workingHours = ls[4]
		place.address = ls[5]
		place.specialization = ls[6]
		place.phone = ls[7]
		place.hasAlcohol = ls[8]
		place.hasDelivery = ls[9]
		place.hasPark = ls[10]
		place.coordinates = ls[11]
		return place

	@staticmethod
	def getMapsCoord(place) -> str:
		"""Insert place or its coordinates and return the link to it on GOOGLE Maps"""
		x = place.coordinates.split(", ")[0]
		y = place.coordinates.split(", ")[1]
		return f"https://www.google.com/maps/search/{x},+{y}/@{x},{y}"

	@staticmethod
	def removePlace(_id: int = 0, coordinates: str = ""):
		"""Query db to remove a place by its id OR coordinates"""
		if _id:
			db.query(f"DELETE FROM places WHERE id = {_id};", True)
		else:
			db.query(f"DELETE FROM places WHERE coordinates = '{coordinates}';", True)

	@staticmethod
	def getPlace(_id: int = None, coord: str = None):
		"""Get a specific place by its id OR by its coordinates"""
		if not _id and not coord:
			raise Exception("ID or coordinates should pe specified")

		if _id:
			return Place.fromList(db.query(f"SELECT * FROM places WHERE id = {_id};"))
		else:
			return Place.fromList(db.query(f"SELECT * FROM places WHERE coordinates = '{coord}';"))

	@staticmethod
	def getAllPlaces() -> list:
		"""Not recommended since it will take long time"""
		db.query("SELECT * FROM places;")
		places = []

		for p in db.lastQuery:
			places.append(Place.fromList(p))

		return places

	@staticmethod
	def getPlacesBySector(sector: str) -> list:
		"""Get all places from the sector"""
		db.query(f"SELECT * FROM places WHERE sector = '{sector}';")
		places = []

		for p in db.lastQuery:
			places.append(Place.fromList(p))

		return places

	@staticmethod
	def setRating(rating: float, *, _id: int = None, place = None):
		"""Set rating of a place"""
		if not _id and not place:
			raise Exception("ID or place object must be supplied")

		if rating < 0 or rating > 5:
			raise Exception(f"Rating can't be {rating}")

		if _id:
			db.query(f"UPDATE places SET rating = {rating} WHERE id = {_id};", True)
		else:
			db.query(f"UPDATE places SET rating = {rating} WHERE id = {place.id};", True)
			place.rating = rating

	@staticmethod
	def setPhone(nr: str, *, _id: int = None, place = None):
		"""Set phone number of a place"""
		if not _id and not place:
			raise Exception("ID or place object must be supplied")

		if len(nr) != 11:
			raise Exception(f"Wrong number length {len(nr)}")

		if _id:
			db.query(f"UPDATE places SET phone = '{nr}' WHERE id = {_id};", True)
		else:
			db.query(f"UPDATE places SET phone = '{nr}' WHERE id = {place.id};", True)
			place.phone = nr

	@staticmethod
	def setSector(sector: str, *, _id: int = None, place = None):
		"""Set sector of a place"""
		if not _id and not place:
			raise Exception("ID or place object must be supplied")

		if _id:
			db.query(f"UPDATE places SET sector = '{sector}' WHERE id = {_id};", True)
		else:
			db.query(f"UPDATE places SET sector = '{sector}' WHERE id = {place.id};", True)
			place.sector = sector

	@staticmethod
	def setSpecialization(spec: str, *, _id: int = None, place = None):
		"""Set specialization of a place"""
		if not _id and not place:
			raise Exception("ID or place object must be supplied")

		if _id:
			db.query(f"UPDATE places SET specialization = '{spec}' WHERE id = {_id};", True)
		else:
			db.query(f"UPDATE places SET specialization = '{spec}' WHERE id = {place.id};", True)
			place.specialization = spec

	@staticmethod
	def setAddress(address: str, *, _id: int = None, place = None):
		"""Set address of a place"""
		if not _id and not place:
			raise Exception("ID or place object must be supplied")

		if _id:
			db.query(f"UPDATE places SET address = '{address}' WHERE id = {_id};", True)
		else:
			db.query(f"UPDATE places SET address = '{address}' WHERE id = {place.id};", True)
			place.address = address

	@staticmethod
	def setWorkingHours(hrs: str, *, _id: int = None, place = None):
		"""Set working hours of a place"""
		if not _id and not place:
			raise Exception("ID or place object must be supplied")

		if _id:
			db.query(f"UPDATE places SET workingHours = '{hrs}' WHERE id = {_id};", True)
		else:
			db.query(f"UPDATE places SET workingHours = '{hrs}' WHERE id = {place.id};", True)
			place.workingHours = hrs

	@staticmethod
	def setCoordinates(coord: str, *, _id: int = None, place = None):
		"""Set Google Maps coordinates of a place"""
		if not _id and not place:
			raise Exception("ID or place object must be supplied")

		if _id:
			db.query(f"UPDATE places SET coordinates = '{coord}' WHERE id = {_id};", True)
		else:
			db.query(f"UPDATE places SET coordinates = '{coord}' WHERE id = {place.id};", True)
			place.coordinates = coord

	@staticmethod
	def setName(name: str, *, _id: int = None, place = None):
		"""Set name of a place"""
		if not _id and not place:
			raise Exception("ID or place object must be supplied")

		if len(name) == 0:
			raise Exception(f"Name can't be empty")

		if _id:
			db.query(f"UPDATE places SET name = '{name}' WHERE id = {_id};", True)
		else:
			db.query(f"UPDATE places SET name = '{name}' WHERE id = {place.id};", True)
			place.name = name

	@staticmethod
	def setDelivery(dev: bool, *, _id: int = None, place = None):
		"""Set whether place has a delivery system"""
		if not _id and not place:
			raise Exception("ID or place object must be supplied")

		if _id:
			db.query(f"UPDATE places SET hasDelivery = {dev} WHERE id = {_id};", True)
		else:
			db.query(f"UPDATE places SET hasDelivery = {dev} WHERE id = {place.id};", True)
			place.hasDelivery = dev

	@staticmethod
	def setParking(park: bool, *, _id: int = None, place = None):
		"""Set whether place has parking slots"""
		if not _id and not place:
			raise Exception("ID or place object must be supplied")

		if _id:
			db.query(f"UPDATE places SET hasPark = {park} WHERE id = {_id};", True)
		else:
			db.query(f"UPDATE places SET hasPark = {park} WHERE id = {place.id};", True)
			place.hasPark = park

	@staticmethod
	def setAlcohol(alc: bool, *, _id: int = None, place = None):
		"""Set whether place has alcohol in stock"""
		if not _id and not place:
			raise Exception("ID or place object must be supplied")

		if _id:
			db.query(f"UPDATE places SET hasAlcohol = {alc} WHERE id = {_id};", True)
		else:
			db.query(f"UPDATE places SET hasAlcohol = {alc} WHERE id = {place.id};", True)
			place.hasAlcohol = alc

	@staticmethod
	def addPlace(place):
		"""Add place to database"""
		query = f'''INSERT INTO places VALUES(
			NULL,
			"{place.sector}",
			"{place.name}",
			{place.rating},
			"{place.workingHours}",
			"{place.address}",
			"{place.specialization}",
			"{place.phone}",
			{place.hasAlcohol},
			{place.hasDelivery},
			{place.hasPark},
			"{place.coordinates}"
		);
		'''

		db.query(query, True)
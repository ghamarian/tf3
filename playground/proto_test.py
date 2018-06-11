from playground.dest import addressbook_pb2

addressbook = addressbook_pb2.AddressBook()
person = addressbook.people.add()
person.id = 1234
person.name = "John doe"
person.email = "a@b.com"
phone = person.phones.add()
phone.number = "111-233"
phone.type = addressbook_pb2.Person.HOME

values = addressbook.SerializeToString()
print(values)

newbook = addressbook_pb2.AddressBook()
newbook.ParseFromString(values)
for person in newbook.people:
    print(person.name)

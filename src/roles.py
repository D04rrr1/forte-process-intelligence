class RoleAssigner:
    def __init__(self):
        # Сколько раз каждый человек
        # находился в каждой зоне
        self.stats = {}

    def update(self, person_id, zones):
        if person_id not in self.stats:
            self.stats[person_id] = {
                "CASHIER_ZONE": 0,
                "CUSTOMER_ZONE": 0,
                "SECURE_ZONE": 0,
                "TOTAL": 0
            }

        self.stats[person_id]["TOTAL"] += 1

        for zone in zones:
            if zone in self.stats[person_id]:
                self.stats[person_id][zone] += 1

    def get_role(self, person_id):
        if person_id not in self.stats:
            return "UNKNOWN"

        data = self.stats[person_id]

        total = data["TOTAL"]

        if total == 0:
            return "UNKNOWN"

        cashier_ratio = (
            data["CASHIER_ZONE"] / total
        )

        customer_ratio = (
            data["CUSTOMER_ZONE"] / total
        )

        secure_ratio = (
            data["SECURE_ZONE"] / total
        )

        # Человек большую часть времени
        # работает внутри кассовой зоны
        if cashier_ratio >= 0.40:
            return "CASHIER"

        # Человек большую часть времени
        # находится перед кассой
        if customer_ratio >= 0.30:
            return "CUSTOMER"

        # Кто-то находится в secure zone,
        # но не похож на кассира
        if secure_ratio >= 0.30:
            return "OTHER_SECURE"

        return "OTHER"

    def summary(self):
        result = {}

        for person_id in self.stats:
            result[person_id] = {
                "role": self.get_role(person_id),
                "stats": self.stats[person_id]
            }

        return result
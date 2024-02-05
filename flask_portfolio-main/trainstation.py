class Graph:
    def __init__(self):
        self.graph = {}

    def add_edge(self, station1, station2, weight):
        if station1 not in self.graph:
            self.graph[station1] = []
        if station2 not in self.graph:
            self.graph[station2] = []
        
        self.graph[station1].append((station2, weight))
        self.graph[station2].append((station1, weight))


def shortest_path(graph, start_station, end_station):
    visited = set()
    queue = [(start_station, [start_station], 0)]  

    while queue:
        current_station, path_taken, num_stations_traveled = queue.pop(0)
        visited.add(current_station)

        if current_station == end_station:
            return path_taken, num_stations_traveled

        for neighbor, _ in graph.get(current_station, []):
            if neighbor not in visited:
                queue.append((neighbor, path_taken + [neighbor], num_stations_traveled + 1))

    return [] 

train_network = Graph()

# LRT Line 1 connections
train_network.add_edge("Roosevelt", "Balintawak", 1)
train_network.add_edge("Balintawak", "Monumento", 1)
train_network.add_edge("Monumento", "5th Avenue", 1)
train_network.add_edge("5th Avenue", "R.Papa", 1)
train_network.add_edge("R.Papa", "Abad Santos", 1)
train_network.add_edge("Abad Santos", "Blumentritt", 1)
train_network.add_edge("Blumentritt", "Tayuman", 1)
train_network.add_edge("Tayuman", "Bambang", 1)
train_network.add_edge("Bambang", "Doroteo Jose", 1)
train_network.add_edge("Doroteo Jose", "Carriedo", 1)
train_network.add_edge("Carriedo", "Central Terminal", 1)
train_network.add_edge("Central Terminal", "United Nations", 1)
train_network.add_edge("United Nations", "Pedro Gil", 1)
train_network.add_edge("Pedro Gil", "Quirino", 1)
train_network.add_edge("Quirino", "Vito Cruz", 1)
train_network.add_edge("Vito Cruz", "Gil Puyat", 1)
train_network.add_edge("Gil Puyat", "Libertad", 1)
train_network.add_edge("Libertad", "EDSA", 1)
train_network.add_edge("EDSA", "Baclaran", 1)

# LRT Line 2 connections
train_network.add_edge("Recto", "Legarda", 1)
train_network.add_edge("Legarda", "Pureza", 1)
train_network.add_edge("Pureza", "V.Mapa", 1)
train_network.add_edge("V.Mapa", "J.Ruiz", 1)
train_network.add_edge("J.Ruiz", "Gilmore", 1)
train_network.add_edge("Gilmore", "Betty Go-Belmonte", 1)
train_network.add_edge("Betty Go-Belmonte", "Araneta-Cubao", 1)
train_network.add_edge("Araneta-Cubao", "Anonas", 1)
train_network.add_edge("Anonas", "Katipunan", 1)
train_network.add_edge("Katipunan", "Santolan", 1)
train_network.add_edge("Santolan", "Marikina", 1)
train_network.add_edge("Marikina", "Antipolo", 1)

# MRT Line 3 connections
train_network.add_edge("North Avenue", "Quezon Avenue", 1)
train_network.add_edge("Quezon Avenue", "GMA Kamuning", 1)
train_network.add_edge("GMA Kamuning", "Araneta-Cubao", 1)
train_network.add_edge("Araneta-Cubao", "Santolan-Anapolis", 1)
train_network.add_edge("Santolan-Anapolis", "Ortigas Avenue", 1)
train_network.add_edge("Ortigas Avenue", "Shaw Boulevard", 1)
train_network.add_edge("Shaw Boulevard", "Boni", 1)
train_network.add_edge("Boni", "Guadalupe", 1)
train_network.add_edge("Guadalupe", "Beundia", 1)
train_network.add_edge("Beundia", "Ayala", 1)
train_network.add_edge("Ayala", "Magallanes", 1)
train_network.add_edge("Magallanes", "Taft Avenue", 1)

# Additional connections
train_network.add_edge("Doroteo Jose", "Recto", 1)
train_network.add_edge("EDSA", "Taft Avenue", 1)
train_network.add_edge("Araneta-Cubao", "Araneta-Cubao", 1)



import random
import collections

class GeneticAlgorithm(object):
    def __init__(
            self,
            max_gens=None,
            population=None,
            mutation_rate=1,
            selection=None,
            circuit_size=10
    ):
        self.max_gens = max_gens
        self.population = population
        self.mutation_rate = mutation_rate
        self.circuit_size = circuit_size
        self.parentGen = []
        self.finalGen = None
        self.currentGen = []
        self.optimumSolution = None
        self.selection = selection

    def generateParents(self, iput_list=None):
        for individual in range(self.population):
            solution = Solution()
            circuit_size = random.randint(len(iput_list), 30)
            solution.generateCircuit(
                no_of_connections=circuit_size,
                iput_list=iput_list
            )
            self.parentGen.append(solution)
            self.currentGen.append(solution)

    def tournamentSelection(self, previous_gen):
        select_amount = random.randint(2, len(previous_gen) - 1)
        selected_individuals = []

        for i in range(select_amount):
            individual = previous_gen[random.randint(0, len(previous_gen) - 1)]

            while individual in selected_individuals:
                individual = previous_gen[random.randint(0, len(previous_gen) - 1)]

            selected_individuals.append(
                individual
            )
        selected_individuals.sort(key=lambda individual: individual.fitness)
        parentA = selected_individuals[0]
        parentB = selected_individuals[1]

        return parentA, parentB



    def start(self, training_inputs, training_outputs):
        gens = []
        total_fitness_per_gen = []
        self.generateParents(iput_list=training_inputs[0])
        self.calculateGenerationFitness(training_inputs, training_outputs)
        previous_gen = self.currentGen
        most_passes = 0
        median_connections_per_gen = []
        for gen in range(self.max_gens):
            self.currentGen = []
            while len(self.currentGen) < self.population:
                if self.selection == 'roulette':
                    parentA, parentB = self.rouletteWheel(previous_gen=previous_gen)
                elif self.selection == 'tournament':
                    parentA, parentB = self.tournamentSelection(previous_gen=previous_gen)
                childA, childB = self.crossover(parentA=parentA, parentB=parentB)
                self.currentGen.append(childA)
                self.currentGen.append(childB)
            for individual in self.currentGen:
                self.mutate(circuit=individual.circuit)
            total_fitness, highest_passed, median_connections = self.calculateGenerationFitness(training_inputs,
                                                                                                training_outputs)
            if highest_passed > most_passes:
                most_passes = highest_passed
            previous_gen = self.currentGen
            total_fitness_per_gen.append(total_fitness)
            median_connections_per_gen.append(median_connections)
            gens.append(gen)
        print()
        self.finalGen = self.currentGen

        return gen, total_fitness_per_gen, gens, highest_passed, median_connections_per_gen

    def test(self, test_set_iputs, test_set_outputs):
        pass

    def calculateGenerationFitness(self, training_inputs, training_outputs):
        fitness = 0
        highest_passed = 0
        median_connections = 0
        for individual in self.currentGen:
            individual.calculateFitness(
                iputs=training_inputs,
                outputs=training_outputs
            )
            if individual.passed > highest_passed:
                highest_passed = individual.passed
            fitness += individual.fitness
            median_connections += len(individual.circuit.connections)
        median_connections = median_connections / len(self.currentGen)
        return fitness, highest_passed, median_connections

    def mutate(self, circuit):
        for i in range(len(circuit.connection_matrix)):
            matrix = circuit.connection_matrix[i]
            conn_index = matrix.connection_number
            pinA_index = matrix.pinA
            pinB_index = matrix.pinB
            gate_index = matrix.gate
            connection_mutation_rate = random.random()
            gate_mutation_rate = random.random()
            if connection_mutation_rate >= self.mutation_rate:
                if gate_mutation_rate >= self.mutation_rate:
                    gate_index = random.randint(0, len(circuit.gates) - 1)
                gate = circuit.gates[gate_index]
                pinA_index = random.randint(0, conn_index - 1)
                pinB_index = random.randint(0, conn_index - 1)
                if gate.label == 'NOT':
                    pinB_index = None
            circuit.connection_matrix[i] = ConnectionMatrix(
                connection_number=conn_index,
                pinA=pinA_index,
                pinB=pinB_index,
                gate=gate_index
            )

        circuit.reCreate()
        circuit.generateMap()

    def crossover(self, parentA, parentB):
        pA_connections = parentA.circuit.connections
        pA_matrix = parentA.circuit.connection_matrix
        pA_iput_conns = parentA.circuit.iput_conns
        pB_connections = parentB.circuit.connections
        pB_matrix = parentB.circuit.connection_matrix
        pB_iput_conns = parentB.circuit.iput_conns
        if len(pA_connections) < len(pB_connections):
            smaller_parent_con = pA_connections
        else:
            smaller_parent_con = pB_connections
        smaller_parent_matrix = None
        if len(pA_matrix) < len(pB_matrix):
            smaller_parent_matrix = pA_matrix
        else:
            smaller_parent_matrix = pB_matrix
        crossover_point = random.randint(1, len(smaller_parent_matrix) - 1)
        c1_select = pA_connections[:crossover_point] + pB_connections[crossover_point:]
        c2_select = pB_connections[:crossover_point] + pA_connections[crossover_point:]
        crossover_point = random.randint(0, len(smaller_parent_matrix) - 1)
        c1_matrix = pA_matrix[:crossover_point] + pB_matrix[crossover_point:]
        c2_matrix = pB_matrix[:crossover_point] + pA_matrix[crossover_point:]
        c1_conns = []
        c2_conns = []
        c1_iput_conns = []
        c2_iput_conns = []

        for i in range(len(parentA.circuit.iput_conns)):
            c1_conn = Connection()
            c2_conn = Connection()

            c1_iput_conns.append(c1_conn)
            c1_conns.append(c1_conn)

            c2_iput_conns.append(c2_conn)
            c2_conns.append(c2_conn)

        for i in range(len(c1_select)):
            if not ((c1_select[i] in pA_iput_conns) or (c1_select[i] in pB_iput_conns)):
                p_conn1 = c1_select[i]
                c1_conn = Connection(gate=p_conn1.gate)
                c1_conns.append(c1_conn)

        for i in range(len(c2_select)):
            if not ((c2_select[i] in pA_iput_conns) or (c2_select[i] in pB_iput_conns)):
                p_conn2 = c2_select[i]
                c2_conn = Connection(gate=p_conn2.gate)
                c2_conns.append(c2_conn)

        c1_circuit = Circuit(
            gates=parentA.circuit.gates,
            iput_conns=c1_iput_conns,
            connections=c1_conns,
            connection_matrix=c1_matrix
        )

        c1_circuit.reCreate()
        self.mutate(c1_circuit)
        c1_circuit.generateMap()

        c2_circuit = Circuit(
            gates=parentB.circuit.gates,
            iput_conns=c2_iput_conns,
            connections=c2_conns,
            connection_matrix=c2_matrix
        )
        c2_circuit.reCreate()
        self.mutate(c2_circuit)
        c2_circuit.generateMap()

        s1 = Solution(circuit=c1_circuit)
        s2 = Solution(circuit=c2_circuit)

        return s1, s2


class Gate(object):
    def __init__(self):
        self.output = None


class AND(Gate):
    def __init__(self):
        self.label = 'AND'
        self.output = None

    def __str__(self):
        return f'Gate(type=AND)'

    def __repr__(self):
        return str(self)

    def do(self, i1, i2):
        if(i1 == 1 and i2 == 1):
            self.output = 1
        else:
            self.output = 0
        return self.output


class OR(Gate):
    def __init__(self):
        self.label = 'OR'
        self.output = None

    def __str__(self):
        return f'Gate(type=OR, output={self.output})'

    def __repr__(self):
        return str(self)

    def do(self, i1, i2):
        if (i1 == 0 and i2 == 0):
            self.output = 0
        else:
            self.output = 1
        return self.output


class XOR(Gate):
    def __init__(self):
        self.label = 'XOR'
        self.output = None

    def __str__(self):
        return f'Gate(type=XOR, output={self.output})'

    def __repr__(self):
        return str(self)

    def do(self, i1, i2):
        if((i1 == 1 and i2 == 1) or (i1 == 0 and i2 == 0)):
            self.output = 0
        else:
            self.output = 1
        return self.output


class NOT(Gate):
    def __init__(self):
        self.label = 'NOT'
        self.output = None

    def __str__(self):
        return f'Gate(type=NOT, output={self.output})'

    def __repr__(self):
        return str(self)

    def do(self, i1):
        if i1 == 1:
            self.output = 0
        else:
            self.output = 1
        return self.output


class NAND(Gate):
    def __init__(self):
        self.label = 'NAND'
        self.output = None

    def __str__(self):
        return f'Gate(type=NAND, output={self.output})'

    def __repr__(self):
        return str(self)

    def do(self, i1, i2):
        if(i1 == 1 and i2 == 1):
            self.output = 0
        else:
            self.output = 1
        return self.output


class NOR(Gate):
    def __init__(self):
        self.label = 'NOR'
        self.output = None

    def __str__(self):
        return f'Gate(type=NOR, output={self.output})'

    def do(self, i1, i2):
        if(i1 == 0 and i2 == 0):
            self.output = 1
        else:
            self.output = 0
        return self.output

class Connection():
    def __init__(self, pinA=None, pinB=None, pinO=None, gate=None, connections=None):
        if not connections:
            self.pinA = pinA
            self.pinB = pinB
        else:
            self.pinA = connections[pinA]
            self.pinB = connections[pinB]
        self.pinO = pinO
        self.gate = gate

    def __str__(self):
        return f'Connection(pinA={self.pinA}, pinB={self.pinB}, pinO={self.pinO}, gate={self.gate})'

    def __repr__(self):
        return str(self)

    def activate(self):
        if self.gate.label == 'NOT':
            self.pinO = self.gate.do(self.pinA.pinO)
        else:
            self.pinO = self.gate.do(self.pinA.pinO, self.pinB.pinO)
        return self.pinO

    def setIput(self, iput):
        self.pinO = iput

    def reset(self):
        self.pinA = None
        self.pinB = None
        self.gate = None
        self.pinO = None

    def edit(self, pinA=None, pinB=None, gate=None):
        self.pinA = pinA
        self.pinB = pinB
        self.gate = gate

ConnectionMatrix = collections.namedtuple(
    'ConnectionMatrix',
    'connection_number pinA pinB gate'
)


class Circuit():
    def __init__(
        self,
        gates=None,
        iput_conns=None,
        connections=None,
        connection_map=None,
        connection_matrix=None
    ):
        self.gates = gates
        self.connections = connections
        self.connection_map = connection_map
        self.iput_conns = iput_conns
        self.connection_matrix = connection_matrix

    def __str__(self):
        return str(self.connection_map)

    def __repr__(self):
        return str(self)

    def reCreate(self):
        for matrix in self.connection_matrix:
            if matrix.pinB is not None:
                self.connections[matrix.connection_number].edit(
                    pinA=self.connections[matrix.pinA],
                    pinB=self.connections[matrix.pinB],
                    gate=self.gates[matrix.gate]
                )
            else:
                self.connections[matrix.connection_number].edit(
                    pinA=self.connections[matrix.pinA],
                    gate=self.gates[matrix.gate]
                )

    def generateMap(self):
        self.connection_map = []
        for connection in self.connections:
            pinA = connection.pinA
            pinB = connection.pinB
            gate = connection.gate
            if not gate:
                mapper = f'c{self.connections.index(connection)}'
                self.connection_map.append(mapper)
            elif gate.label == 'NOT':
                mapper = f'c{self.connections.index(connection)}: c{self.connections.index(pinA)} to {gate.label} gate'
                self.connection_map.append(mapper)
            else:
                mapper = f'c{self.connections.index(connection)}: c{self.connections.index(pinA)} + c{self.connections.index(pinB)} to {gate.label} gate'
                self.connection_map.append(mapper)

    def generateMatrix(self):
        self.connection_matrix = []
        for connection in self.connections:
            if connection.gate:
                if connection.pinB:
                    matrix = ConnectionMatrix(
                        connection_number=self.connections.index(connection),
                        pinA=self.connections.index(connection.pinA),
                        pinB=self.connections.index(connection.pinB),
                        gate=self.gates.index(connection.gate)
                    )
                else:
                    matrix = ConnectionMatrix(
                        connection_number=self.connections.index(connection),
                        pinA=self.connections.index(connection.pinA),
                        pinB=None,
                        gate=self.gates.index(connection.gate)
                    )
                self.connection_matrix.append(matrix)
    def activate(self):
        if self.connections:
            for connection in self.connections:
                if connection.gate:
                    connection.activate()
            output = self.connections[-1].pinO
            return output
        else:
            return 0
    def setIputs(self, iput_list):
        for iput in iput_list:
            self.iput_conns[iput_list.index(iput)].setIput(iput=iput)



class Solution(object):
    def __init__(self, circuit=None):
        self.circuit = circuit
        self.outputs = []
        self.passed = 0
        self.simplicity = 0
        self.fitness = 0
        self.MAX_CONNECTIONS = 10

    def test(self, training_set):
        pass

    def generateCircuit(self, no_of_connections, iput_list):
        and_gate = AND()
        or_gate = OR()
        not_gate = NOT()
        xor_gate = XOR()
        nand_gate = NAND()
        nor_gate = NOR()
        gates = [and_gate, or_gate, not_gate, xor_gate, nand_gate, nor_gate]
        iput_conns = []
        connections = []
        connection_matrix = []
        connection_number = 1
        for iput in iput_list:
            iput_conn = Connection(pinO=iput)
            iput_conns.append(iput_conn)
            connections.append(iput_conn)
        for i in range(len(iput_conns)):
            gate = gates[random.randint(0, len(gates) - 1)]
            if gate.label == 'NOT':
                pinA = iput_conns[random.randint(0, len(iput_conns) - 1)]
                pinB = None
                connection = Connection(
                    pinA=pinA,
                    gate=gate
                )
            else:
                pinA = iput_conns[random.randint(0, len(iput_conns) - 1)]
                pinB = iput_conns[random.randint(0, len(iput_conns) - 1)]
                connection = Connection(
                    pinA=pinA,
                    pinB=pinB,
                    gate=gate
                )
            connections.append(connection)
            new_conn = connections.index(connections[-1])
            matrix = ConnectionMatrix(
                connection_number=new_conn,
                pinA=connections.index(pinA),
                pinB=connections.index(pinB) if pinB else None,
                gate=gates.index(gate)
            )
            connection_matrix.append(matrix)
            no_of_connections -= 1
            connection_number += 1
        for i in range(no_of_connections):
            gate = gates[random.randint(0, len(gates) - 1)]
            if(gate.label == 'NOT'):
                pinA = connections[random.randint(0, len(connections) - 1)]
                pinB = None
                while not pinA.gate:
                    pinA = connections[random.randint(0, len(connections) - 1)]
                connection = Connection(pinA=pinA, gate=gate)
            else:
                pinA = connections[random.randint(0, len(connections) - 1)]
                pinB = connections[random.randint(0, len(connections) - 1)]
                while not (pinA.gate or pinB.gate):
                    pinA = connections[random.randint(0, len(connections) - 1)]
                    pinB = connections[random.randint(0, len(connections) - 1)]
                connection = Connection(pinA=pinA, pinB=pinB, gate=gate)
            connections.append(connection)
            new_conn = connections.index(connections[-1])
            matrix = ConnectionMatrix(
                connection_number=new_conn,
                pinA=connections.index(pinA),
                pinB=connections.index(pinB) if pinB else None,
                gate=gates.index(gate)
            )
            connection_matrix.append(matrix)
            connection_number += 1
        gate = gates[random.randint(0, len(gates) - 1)]
        while(gate.label == 'NOT'):
            gate = gates[random.randint(0, len(gates) - 1)]
        pinA = connections[-1]
        pinB = connections[-2]
        output_conn = Connection(pinA=pinA, pinB=pinB, gate=gate)
        connections.append(output_conn)
        new_conn = connections.index(connections[-1])
        matrix = ConnectionMatrix(
            connection_number=new_conn,
            pinA=connections.index(pinA),
            pinB=connections.index(pinB) if pinB else None,
            gate=gates.index(gate)
        )
        connection_matrix.append(matrix)
        self.circuit = Circuit(
            gates=gates,
            iput_conns=iput_conns,
            connections=connections,
            connection_matrix=connection_matrix
        )
        self.circuit.generateMap()
        return self.circuit

    def calculateFitness(self, iputs, outputs):
        size = len(iputs)
        no_1s = outputs.count(1)
        no_0s = outputs.count(0)
        passed_1s = 0
        passed_0s = 0
        self.passed = 0
        self.outputs = []
        for i in range(size):
            iput = iputs[i]
            expected_output = outputs[i]
            self.circuit.setIputs(iput_list=iput)
            actual_output = self.circuit.activate()
            self.outputs.append(actual_output)
            if actual_output == expected_output:
                self.passed += 1
        passed = 0
        total = 2 ** len(outputs)
        for i in range(len(outputs)):
            desired = outputs[i]
            actual = self.outputs[i]
            if actual == desired:
                passed += (2 ** i)
        pass_percentage = (passed / total) * 100
        no_of_connections = len(self.circuit.connection_matrix)
        fitness = (pass_percentage * no_of_connections)
        self.fitness = fitness
        return self.fitness

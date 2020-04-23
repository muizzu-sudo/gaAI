import matplotlib.pyplot
from FunctionsGA import GeneticAlgorithm

if __name__ == '__main__':
    print()

    try:
        file = open('data-sets/data2.txt')

        contents = []
        input_set = []
        inputs = []
        outputs = []

        contents = file.readlines()

        for content in contents:
            split_contents = content.split(" ")
            output = int(split_contents[-1])
            outputs.append(output)
            split_contents.pop(-1)
            input_set = [round(float(x)) for x in split_contents]
            inputs.append(input_set)

        training_set_inputs = inputs[:len(inputs) // 2]
        training_set_outputs = outputs[:len(outputs) // 2]
        test_set_inputs = inputs[len(inputs) // 2:]
        test_set_outputs = outputs[len(outputs) // 2:]

        print('========== DATA-SET 2 ============')
        print()
        print('DATA-SET_SIZE: 64')
        print('TRAINING_DATA_SIZE : '+ str(len(training_set_outputs)))
        print('TEST_DATA_SIZE : '+ str(len(test_set_outputs)))

        ga = GeneticAlgorithm(
            max_gens=200,
            population=10,
            mutation_rate=0.8,
            selection='tournament'
        )

        (
            no_of_generations,
            fitnesses,
            gens,
            highest,
            median_connections
        ) = ga.start(
            training_inputs=training_set_inputs,
            training_outputs=training_set_outputs
        )

        print("NUMBER OF GENERATIONS: " + str(no_of_generations))

        print()
        print('========== FINAL GENERATION EVOLVED =======================')
        print()
        for individual in ga.finalGen:
            print('Individual: ' + str(individual.circuit))
            print('Fitness: ' + str(individual.fitness))
            print('  Expected Result: ' + str(training_set_outputs))
            print('Calculated Result: ' + str(individual.outputs))
            percentile = 100 / len(test_set_outputs)
            print('Correctly Classified: ' + str(individual.passed) + ' out of 32 ('+ str(individual.passed * percentile) + '%)')
            print()

        matplotlib.pyplot.title('Total Fitness Per Generation', fontsize=20, color='red')
        matplotlib.pyplot.xlabel('Generation', fontsize=20)
        matplotlib.pyplot.ylabel('Fitness', fontsize=20)
        matplotlib.pyplot.plot(gens, fitnesses, linewidth=2, color='black')
        matplotlib.pyplot.draw()
        matplotlib.pyplot.show()

    except Exception as e:
        print('Error!')

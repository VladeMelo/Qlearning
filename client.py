import connection as cn
import numpy as np

s = cn.connect(2037)

ACTIONS = ['left', 'right', 'jump']

DIRECTIONS = {
    '00': 0, # Norte
    '01': 1, # Leste
    '10': 2, # Sul
    '11': 3 # Oeste
}

EPSILON = 0.8 # porcentagem de vezes que devemos executar a melhor ação
DISCOUNT_FACTOR = 0.95 # determina a importância que daremos aos rewards de longo prazo
LEARNING_RATE = 0.7 # taxa na qual o modelo deve aprender

TRAINING_TIMES = 5 # número de vezes que vai treinar o Amongois a chegar até o objetivo

# carrega a tabela
def load_table():
    with open('resultado.txt') as file:
        table = np.loadtxt(file)

    return table

Q_TABLE = load_table()

# salva a tabela
def save_table():
    np.savetxt('resultado.txt', Q_TABLE)

def get_next_action(platform, direction):
    if np.random.random() < EPSILON: # escolhe a melhor ação perante a Q_TABLE para aquele estado
        return np.argmax(Q_TABLE[platform * 4 + DIRECTIONS[direction]])
    else: # escolhe uma ação aleatória
        return np.random.randint(len(ACTIONS))

for episode in range(TRAINING_TIMES):
    platform = 0 # plataforma atual
    direction = '00' # direção atual

    is_finish = False # para checar quando o Amongois chega no objetivo

    while not is_finish:
        # escolhe a próxima ação
        action_index = get_next_action(platform, direction)
    
        # recebe a nova posição e o reward de ter chegado nela
        new_platform, reward = cn.get_state_reward(s, ACTIONS[action_index])

        # calcula a diferença temporal
        previous_q_value = Q_TABLE[platform * 4 + DIRECTIONS[direction], action_index]
        temporal_difference = reward + (DISCOUNT_FACTOR * np.max(Q_TABLE[int(new_platform[:-2], 2) * 4 + DIRECTIONS[new_platform[-2:]]])) - previous_q_value

        # atualiza a Q_TABLE com o novo valor após a ação
        new_q_value = previous_q_value + (LEARNING_RATE * temporal_difference)
        Q_TABLE[platform * 4 + DIRECTIONS[direction], action_index] = new_q_value

        # atualiza o estado da plataforma e da direção que o Amongois possui
        platform = int(new_platform[:-2], 2)
        direction = new_platform[-2:]

        # verifica se chegou no final
        is_finish = reward == 300

        save_table()

    print('Finish Episode') # relata a finalização de um episódio de treino

print('Training complete!') # relata a finalização de todos os episódios de treino, ou seja, do treinamento em si
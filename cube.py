from random import randint
import numpy as np

class Cube:
    def __init__(self, ds, var=0):
        '''
        white == 0
        yellow == 1
        orange == 2
        red == 3
        blue == 4
        green == 5
        '''
        self.ds = ds

        if self.ds == 2:
            self.front = [[0] * 2 for i in range(2)]
            self.back = [[1] * 2 for i in range(2)]
            self.left = [[2] * 2 for i in range(2)]
            self.right = [[3] * 2 for i in range(2)]
            self.up = [[4] * 2 for i in range(2)]
            self.bottom = [[5] * 2 for i in range(2)]
        else:
            self.front = [[0] * 3 for i in range(3)]
            self.back = [[1] * 3 for i in range(3)]
            self.left = [[2] * 3 for i in range(3)]
            self.right = [[3] * 3 for i in range(3)]
            self.up = [[4] * 3 for i in range(3)]
            self.bottom = [[5] * 3 for i in range(3)]

        self.counter = 0
        if var == 0:
            self.make_random_moves(8)
        elif var == 1:
            self.move_front_clockwise()
        elif var == 2:
            self.move_front_counterclockwise()
        self.counter = 0
        self.history = []

    def show(self):
        print(self.front, self.back, self.left, self.right, self.up, self.bottom, sep='\n')

    def get_state(self):
        return np.array([self.front, self.back, self.left, self.right, self.up, self.bottom]).reshape(1,-1)

    def get_history(self):
        return self.history

    def count_layers_done(self):
        layers_done = 0
        for side in [self.front, self.back, self.left, self.right, self.up, self.bottom]:
            for i in range(len(side)):
                layer = 1
                first = side[i][0]
                for j in range(1,len(side)):
                    if side[i][j] == first:
                        layer += 1
                    else:
                        break
                if layer == self.ds:
                    layers_done += 1

                layer = 1
                first = side[0][i]
                for j in range(1, len(side)):
                    if side[j][i] == first:
                        layer += 1
                    else:
                        break
                if layer == self.ds:
                    layers_done += 1

        return layers_done

    def get_side_incorrect_squares(self, side_name):
        incorrect_squares = 0

        if side_name == 'front':
            side = self.front
        elif side_name == 'back':
            side = self.back
        elif side_name == 'left':
            side = self.left
        elif side_name == 'right':
            side = self.right
        elif side_name == 'up':
            side = self.up
        elif side_name == 'bottom':
            side = self.bottom
        else:
            side = self.front

        if self.ds==2:
            colors = np.array([0,0,0,0,0,0])
            for i in range(2):
                for j in range(2):
                    color = side[i][j]
                    colors[color] += 1

            return 4-colors.max()
        else:
            middle_square = side[1][1]

            for i in range(3):
                for j in range(3):
                    if i==1 and j==1:
                        pass
                    incorrect_squares += int(middle_square != side[i][j])

        return incorrect_squares

    # cube entropy is a sum of all squares on the wrong sides divided by its maximum
    def get_cube_entropy(self):
        total_incorrect_squares = 0

        total_incorrect_squares += self.get_side_incorrect_squares('front')
        total_incorrect_squares += self.get_side_incorrect_squares('back')
        total_incorrect_squares += self.get_side_incorrect_squares('up')
        total_incorrect_squares += self.get_side_incorrect_squares('bottom')
        total_incorrect_squares += self.get_side_incorrect_squares('left')
        total_incorrect_squares += self.get_side_incorrect_squares('right')

        if self.ds==2:
            return total_incorrect_squares/24.0
        else:
            return total_incorrect_squares/48.0

    # every important move on cube
    # 4 rolls
    # 10 moves

    # rolling
    def roll_forward(self):
        self.front, self.up, self.back, self.bottom = self.bottom, self.front, self.up, self.back

    def roll_backward(self):
        self.front, self.up, self.back, self.bottom = self.up, self.back, self.bottom, self.front

    def roll_left(self):
        self.left, self.up, self.right, self.bottom = self.up, self.right, self.bottom, self.left

    def roll_right(self):
        self.right, self.up, self.left, self.bottom = self.up, self.left, self.bottom, self.right

    # moving
    def move_front_clockwise(self):
        if self.ds==2:
            (
                self.front[0][0], self.front[0][1],
                self.front[1][0], self.front[1][1])=(
                self.front[1][0], self.front[0][0],
                self.front[1][1], self.front[0][1]
            )

            (
                self.left[0][1], self.left[1][1],
                self.up[1][0], self.up[1][1],
                self.right[0][0], self.right[1][0],
                self.bottom[0][0], self.bottom[0][1],
            ) = (
                self.bottom[0][0], self.bottom[0][1],
                self.left[1][1], self.left[0][1],
                self.up[1][0], self.up[1][1],
                self.right[1][0], self.right[0][0],
            )

        else:
            (
                self.front[0][0], self.front[0][1], self.front[0][2],
                self.front[1][0],                   self.front[1][2],
                self.front[2][0], self.front[2][1], self.front[2][2]) = (

                self.front[2][0], self.front[1][0], self.front[0][0],
                self.front[2][1],                   self.front[0][1],
                self.front[2][2], self.front[1][2], self.front[0][2]
            )


            (
                self.left[0][2], self.left[1][2], self.left[2][2],
                self.up[2][0], self.up[2][1], self.up[2][2],
                self.right[0][0], self.right[1][0], self.right[2][0],
                self.bottom[0][0], self.bottom[0][1], self.bottom[0][2],
                ) = (

                self.bottom[0][0], self.bottom[0][1], self.bottom[0][2],
                self.left[2][2], self.left[1][2], self.left[0][2],
                self.up[2][0], self.up[2][1], self.up[2][2],
                self.right[2][0], self.right[1][0], self.right[0][0]
            )

    def move_front_counterclockwise(self):
        # front squares

        if self.ds==2:

            (
                self.front[0][0], self.front[0][1],
                self.front[1][0], self.front[1][1])=(
                self.front[0][1], self.front[1][1],
                self.front[0][0], self.front[1][0]
            )

            (
                self.left[0][1], self.left[1][1],
                self.up[1][0], self.up[1][1],
                self.right[0][0], self.right[1][0],
                self.bottom[0][0], self.bottom[0][1],
            ) = (

                self.up[1][1], self.up[1][0],
                self.right[0][0], self.right[1][0],
                self.bottom[0][1], self.bottom[0][0],
                self.left[1][1], self.left[0][1],
            )
        else:
            (
                self.front[0][0], self.front[0][1], self.front[0][2],
                self.front[1][0],                   self.front[1][2],
                self.front[2][0], self.front[2][1], self.front[2][2]) = (

                self.front[0][2], self.front[1][2], self.front[2][2],
                self.front[0][1],                 self.front[2][1],
                self.front[0][0], self.front[1][0], self.front[2][0]
            )

            (
                self.left[0][2], self.left[1][2], self.left[2][2],
                self.up[2][0], self.up[2][1], self.up[2][2],
                self.right[0][0], self.right[1][0], self.right[2][0],
                self.bottom[0][0], self.bottom[0][1], self.bottom[0][2],
            ) = (
                self.up[2][2], self.up[2][1], self.up[2][0],
                self.right[0][0], self.right[1][0], self.right[2][0],
                self.bottom[0][2], self.bottom[0][1], self.bottom[0][0],
                self.left[0][2], self.left[1][2], self.left[2][2],
            )

    def move_left_clockwise(self):
        if self.ds==2:
            (
                self.left[0][0], self.left[0][1],
                self.left[1][0], self.left[1][1]) = (
                self.left[1][0], self.left[0][0],
                self.left[1][1], self.left[0][1]
            )

            (
                self.back[0][1], self.back[1][1],
                self.up[1][0], self.up[1][1],
                self.front[0][0], self.front[1][0],
                self.bottom[0][0], self.bottom[0][1],
            ) = (
                self.bottom[0][0], self.bottom[0][1],
                self.back[1][1], self.back[0][1],
                self.up[1][0], self.up[1][1],
                self.front[1][0], self.front[0][0]
            )
        else:
            (
                self.left[0][0], self.left[0][1], self.left[0][2],
                self.left[1][0],                  self.left[1][2],
                self.left[2][0], self.left[2][1], self.left[2][2]) = (

                self.left[2][0], self.left[1][0], self.left[0][0],
                self.left[2][1],                  self.left[0][1],
                self.left[2][2], self.left[1][2], self.left[0][2]
            )

            (
                self.back[0][2], self.back[1][2], self.back[2][2],
                self.up[2][0], self.up[2][1], self.up[2][2],
                self.front[0][0], self.front[1][0], self.front[2][0],
                self.bottom[0][0], self.bottom[0][1], self.bottom[0][2],
            ) = (
                self.bottom[0][0], self.bottom[0][1], self.bottom[0][2],
                self.back[2][2], self.back[1][2], self.back[0][2],
                self.up[2][0], self.up[2][1], self.up[2][2],
                self.front[2][0], self.front[1][0], self.front[0][0]
            )

    def move_left_counterclockwise(self):
        if self.ds==2:
            (
                self.left[0][0], self.left[0][1],
                self.left[1][0], self.left[1][1]) = (
                self.left[0][1], self.left[1][1],
                self.left[0][0], self.left[1][0]
            )

            (
                self.back[0][1], self.back[1][1],
                self.up[1][0], self.up[1][1],
                self.front[0][0], self.front[1][0],
                self.bottom[0][0], self.bottom[0][1],
            ) = (
                self.up[1][1], self.up[1][0],
                self.front[0][0], self.front[1][0],
                self.bottom[0][1], self.bottom[0][0],
                self.back[0][1], self.back[1][1],
            )
        else:
            (
                self.left[0][0], self.left[0][1], self.left[0][2],
                self.left[1][0],                  self.left[1][2],
                self.left[2][0], self.left[2][1], self.left[2][2]) = (

                self.left[0][2], self.left[1][2], self.left[2][2],
                self.left[0][1],                  self.left[2][1],
                self.left[0][0], self.left[1][0], self.left[2][0]
            )

            (
                self.back[0][2], self.back[1][2], self.back[2][2],
                self.up[2][0], self.up[2][1], self.up[2][2],
                self.front[0][0], self.front[1][0], self.front[2][0],
                self.bottom[0][0], self.bottom[0][1], self.bottom[0][2],
            ) = (

                self.up[2][2], self.up[2][1], self.up[2][0],
                self.front[0][0], self.front[1][0], self.front[2][0],
                self.bottom[0][2], self.bottom[0][1], self.bottom[0][0],
                self.back[0][2], self.back[1][2], self.back[2][2],
            )

    def move_right_clockwise(self):
        if self.ds==2:
            (
                self.right[0][0], self.right[0][1],
                self.right[1][0], self.right[1][1]) = (
                self.right[1][0], self.right[0][0],
                self.right[1][1], self.right[0][1]
            )

            (
                self.front[0][1], self.front[1][1],
                self.up[0][1], self.up[1][1],
                self.back[1][0], self.back[0][0],
                self.bottom[1][0], self.bottom[0][0],
            ) = (
                self.bottom[1][0], self.bottom[0][0],
                self.front[0][1], self.front[1][1],
                self.up[0][1], self.up[1][1],
                self.back[1][0], self.back[0][0],
            )
        else:
            (
                self.right[0][0], self.right[0][1], self.right[0][2],
                self.right[1][0],                   self.right[1][2],
                self.right[2][0], self.right[2][1], self.right[2][2]) = (

                self.right[2][0], self.right[1][0], self.right[0][0],
                self.right[2][1],                   self.right[0][1],
                self.right[2][2], self.right[1][2], self.right[0][2]
            )

            (
                self.front[0][2], self.front[1][2], self.front[2][2],
                self.up[0][2], self.up[1][2], self.up[2][2],
                self.back[2][0], self.back[1][0], self.back[0][0],
                self.bottom[2][0], self.bottom[1][0], self.bottom[0][0],
            ) = (

                self.bottom[2][0], self.bottom[1][0], self.bottom[0][0],
                self.front[0][2], self.front[1][2], self.front[2][2],
                self.up[0][2], self.up[1][2], self.up[2][2],
                self.back[2][0], self.back[1][0], self.back[0][0],
            )

    def move_right_counterclockwise(self):
        if self.ds==2:
            (
                self.right[0][0], self.right[0][1],
                self.right[1][0], self.right[1][1]) = (
                self.right[0][1], self.right[1][1],
                self.right[0][0], self.right[1][0]
            )

            (
                self.front[0][1], self.front[1][1],
                self.up[0][1], self.up[1][1],
                self.back[1][0], self.back[0][0],
                self.bottom[1][0], self.bottom[0][0],
            ) = (
                self.up[0][1], self.up[1][1],
                self.back[1][0], self.back[0][0],
                self.bottom[1][0], self.bottom[0][0],
                self.front[0][1], self.front[1][1],
            )
        else:

            (
                self.right[0][0], self.right[0][1], self.right[0][2],
                self.right[1][0],                   self.right[1][2],
                self.right[2][0], self.right[2][1], self.right[2][2]) = (

                self.right[0][2], self.right[1][2], self.right[2][2],
                self.right[0][1],                   self.right[2][1],
                self.right[0][0], self.right[1][0], self.right[2][0]
            )

            (
                self.front[0][2], self.front[1][2], self.front[2][2],
                self.up[0][2], self.up[1][2], self.up[2][2],
                self.back[2][0], self.back[1][0], self.back[0][0],
                self.bottom[2][0], self.bottom[1][0], self.bottom[0][0],
            ) = (

                self.up[0][2], self.up[1][2], self.up[2][2],
                self.back[2][0], self.back[1][0], self.back[0][0],
                self.bottom[2][0], self.bottom[1][0], self.bottom[0][0],
                self.front[0][2], self.front[1][2], self.front[2][2],
            )

    def move_up_clockwise(self):
        if self.ds==2:
            (
                self.up[0][0], self.up[0][1],
                self.up[1][0], self.up[1][1]) = (
                self.up[1][0], self.up[0][0],
                self.up[1][1], self.up[0][1]
            )

            (
                self.left[0][0], self.left[0][1],
                self.back[0][0], self.back[0][1],
                self.right[0][0], self.right[0][1],
                self.front[0][0], self.front[0][1],
            ) = (

                self.front[0][0], self.front[0][1],
                self.left[0][0], self.left[0][1],
                self.back[0][0], self.back[0][1],
                self.right[0][0], self.right[0][1],
            )
        else:
            (
                self.up[0][0], self.up[0][1], self.up[0][2],
                self.up[1][0],                self.up[1][2],
                self.up[2][0], self.up[2][1], self.up[2][2]) = (

                self.up[2][0], self.up[1][0], self.up[0][0],
                self.up[2][1],                self.up[0][1],
                self.up[2][2], self.up[1][2], self.up[0][2]
            )

            (
                self.left[0][0], self.left[0][1], self.left[0][2],
                self.back[0][0], self.back[0][1], self.back[0][2],
                self.right[0][0], self.right[0][1], self.right[0][2],
                self.front[0][0], self.front[0][1], self.front[0][2],
            ) = (

                self.front[0][0], self.front[0][1], self.front[0][2],
                self.left[0][0], self.left[0][1], self.left[0][2],
                self.back[0][0], self.back[0][1], self.back[0][2],
                self.right[0][0], self.right[0][1], self.right[0][2],
            )

    def move_up_counterclockwise(self):
        if self.ds==2:
            (
                self.up[0][0], self.up[0][1],
                self.up[1][0], self.up[1][1]) = (
                self.up[0][1], self.up[1][1],
                self.up[0][0], self.up[1][0]
            )

            (
                self.left[0][0], self.left[0][1],
                self.back[0][0], self.back[0][1],
                self.right[0][0], self.right[0][1],
                self.front[0][0], self.front[0][1],
            ) = (
                self.back[0][0], self.back[0][1],
                self.right[0][0], self.right[0][1],
                self.front[0][0], self.front[0][1],
                self.left[0][0], self.left[0][1],
            )
        else:

            (
                self.up[0][0], self.up[0][1], self.up[0][2],
                self.up[1][0],                self.up[1][2],
                self.up[2][0], self.up[2][1], self.up[2][2]) = (

                self.up[0][2], self.up[1][2], self.up[2][2],
                self.up[0][1],                self.up[2][1],
                self.up[0][0], self.up[1][0], self.up[2][0]
            )

            (
                self.left[0][0], self.left[0][1], self.left[0][2],
                self.back[0][0], self.back[0][1], self.back[0][2],
                self.right[0][0], self.right[0][1], self.right[0][2],
                self.front[0][0], self.front[0][1], self.front[0][2],
            ) = (

                self.back[0][0], self.back[0][1], self.back[0][2],
                self.right[0][0], self.right[0][1], self.right[0][2],
                self.front[0][0], self.front[0][1], self.front[0][2],
                self.left[0][0], self.left[0][1], self.left[0][2],
            )

    def move_bottom_clockwise(self):
        if self.ds ==2:
            (
                self.bottom[0][0], self.bottom[0][1],
                self.bottom[1][0], self.bottom[1][1]) = (
                self.bottom[1][0], self.bottom[0][0],
                self.bottom[1][1], self.bottom[0][1]
            )

            (
                self.left[1][0], self.left[1][1],
                self.back[1][0], self.back[1][1],
                self.right[1][0], self.right[1][1],
                self.front[1][0], self.front[1][1],
            ) = (
                self.front[1][0], self.front[1][1],
                self.left[1][0], self.left[1][1],
                self.back[1][0], self.back[1][1],
                self.right[1][0], self.right[1][1],
            )
        else:

            (
                self.bottom[0][0], self.bottom[0][1], self.bottom[0][2],
                self.bottom[1][0], self.bottom[1][2],
                self.bottom[2][0], self.bottom[2][1], self.bottom[2][2]) = (

                self.bottom[2][0], self.bottom[1][0], self.bottom[0][0],
                self.bottom[2][1], self.bottom[0][1],
                self.bottom[2][2], self.bottom[1][2], self.bottom[0][2]
            )

            (
                self.left[2][0], self.left[2][1], self.left[2][2],
                self.back[2][0], self.back[2][1], self.back[2][2],
                self.right[2][0], self.right[2][1], self.right[2][2],
                self.front[2][0], self.front[2][1], self.front[2][2],
            ) = (

                self.front[2][0], self.front[2][1], self.front[2][2],
                self.left[2][0], self.left[2][1], self.left[2][2],
                self.back[2][0], self.back[2][1], self.back[2][2],
                self.right[2][0], self.right[2][1], self.right[2][2],
            )

    def move_bottom_counterclockwise(self):
        if self.ds==2:
            (
                self.bottom[0][0], self.bottom[0][1],
                self.bottom[1][0], self.bottom[1][1]) = (
                self.bottom[0][1], self.bottom[1][1],
                self.bottom[0][0], self.bottom[1][0]
            )

            (
                self.left[1][0], self.left[1][1],
                self.back[1][0], self.back[1][1],
                self.right[1][0], self.right[1][1],
                self.front[1][0], self.front[1][1],
            ) = (
                self.back[1][0], self.back[1][1],
                self.right[1][0], self.right[1][1],
                self.front[1][0], self.front[1][1],
                self.left[1][0], self.left[1][1],
            )
        else:

            (
                self.bottom[0][0], self.bottom[0][1], self.bottom[0][2],
                self.bottom[1][0], self.bottom[1][2],
                self.bottom[2][0], self.bottom[2][1], self.bottom[2][2]) = (

                self.bottom[0][2], self.bottom[1][2], self.bottom[2][2],
                self.bottom[0][1], self.bottom[2][1],
                self.bottom[0][0], self.bottom[1][0], self.bottom[2][0]
            )

            (
                self.left[2][0], self.left[2][1], self.left[2][2],
                self.back[2][0], self.back[2][1], self.back[2][2],
                self.right[2][0], self.right[2][1], self.right[2][2],
                self.front[2][0], self.front[2][1], self.front[2][2],
            ) = (

                self.back[2][0], self.back[2][1], self.back[2][2],
                self.right[2][0], self.right[2][1], self.right[2][2],
                self.front[2][0], self.front[2][1], self.front[2][2],
                self.left[2][0], self.left[2][1], self.left[2][2],
            )

    def make_move(self, move):
        if move == 0:
            self.roll_forward()
        elif move == 1:
            self.roll_backward()
        elif move == 2:
            self.roll_left()
        elif move == 3:
            self.roll_right()
        elif move == 4:
            self.move_front_clockwise()
        elif move == 5:
            self.move_front_counterclockwise()
        elif move == 6:
            self.move_left_clockwise()
        elif move == 7:
            self.move_left_counterclockwise()
        elif move == 8:
            self.move_right_clockwise()
        elif move == 9:
            self.move_right_counterclockwise()
        elif move == 10:
            self.move_up_clockwise()
        elif move == 11:
            self.move_up_counterclockwise()
        elif move == 12:
            self.move_bottom_clockwise()
        elif move == 13:
            self.move_bottom_counterclockwise()

        self.history.append(move)

        self.counter += 1
        return self.get_state(), self.get_cube_entropy(), self.counter, self.get_cube_entropy()==0, self.counter>=100

    def make_random_moves(self, n):
        for i in range(n):
            rand = randint(0,14)
            self.make_move(rand)
# How to use Self-Testing Mode

Self-testing mode is a feature that can be used in any dpvis program to test your knowledge of a program's execution. For the entirety of this tutotial, we will use the Knapsack example.

After running the code for Knapsack, the following window will show up.

![alt text](/docs/test_mode/images/start_screen.png "Title")

<img src="../images/start_screen.png" width="75%"/>

On the left is a gray panel containing the test mode controls. There is a "Test Myself!" button and checkboxes. The button can be used to switch to testing mode. In testing mode, there are three types of tests, write tests, read tests, and value tests, which can be turned off by unchecking the corresponding checkbox.

![alt text](/docs/test_mode/images/checkboxes.png "Title")

<img src="../images/checkboxes.png" width="75%"/>

For this tutorial, we will perform each type of test. To start press the "Test Myself!" button and the screen should change to the following (it may take a second for the window to load):

![alt text](/docs/test_mode/images/test_mode.png "Title")

<img src="../images/test_mode.png" width="75%"/>

Note that the red alert under the "Test Myself!" button tells us what test we are on. Since it says,  "what cells are written to in the next frame?" we are on the write test.

## Write Tests
This test can be turned off by unchecking the "What is the next cell?" checkbox before switching to testing mode.

The goal of the write test is to select the cells that would be written to in the next timestep (i.e. what cells would be <span
style="background-color:#5c53a5">purple</span>). A cell can be selected as your answer by clicking it with your mouse pointer. To practice, let's try clicking an incorrect cell. Use your mouse pointer to click cell in row "Item 2: (4, 3)" and column "Capacity 0".

![alt text](/docs/test_mode/images/wrong_write.png "Title")

<img src="../images/wrong_write.png" width="75%"/>

On the left panel some feeback will show up saying you selected the incorrect cell. Now try clicking the correct cell (row "Item 1: (2, 4)" and column "Capacity 1").

![alt text](/docs/test_mode/images/correct_write.png "Title")

<img src="../images/correct_write.png" width="75%"/>

The cell selected cell will turn <span
style="background-color:#5c53a5">purple</span> because it was correct. Since there was only a single cell written to on the next timestep, we completed the write test. The test mode will prompt you for the Read test. Note that if there were multiple cells written to we would have to continue clicking on cells.

## Read Tests
This test can be turned off by unchecking the "What are its dependencies?" checkbox before switching to testing mode.

In the read test, our goal is to select all cells that were read from in the next timestep (would be <span
style="background-color:#b7609a">pink</span>). Just like in the write test, you can select cells as your answer by clicking on them. Try answering the question on your own by clicking cells (hint: use the recurrence on the top left of the screen to see what cells are read from)

![alt text](/docs/test_mode/images/recurrence.png "Title")

<img src="../images/recurrence.png" width="75%"/>

Recall from the write test that we wrote to cell `i=1` and `C=1`. As shown in the recurrence, when this write occurs, we access `OPT(0, 1)` and `OPT(0, 1 - c(1))`. Since item `i=1` has capacity `c(1) = 2`, we can ignore the second term because we cannot access a cell with negative capacity. Thus, we only access `OPT(0, 1)`. If you did not already click on the correct cell, click on it now (i.e. the cell at row "Item 1: (2, 4)" and "Capacity 1").

![alt text](/docs/test_mode/images/correct_read.png "Title")

<img src="../images/correct_read" width="75%"/>

The cell will turn <span
style="background-color:#b7609a">pink</span> since it was the correct answer. Since only one cell was read from, we completed the read test and the test mode will prompt you for the value test. Note that if more cells were read from, they would have to be clicked on before moving on to the next test.

## Value Tests
This test can be turned off by unchecking the "What is its value?" checkbox before switching to testing mode.

In the value test, you will have to enter the value written to all cells on the next timestep. The tester will highlight a single <span
style="background-color:#5c53a5">purple</span> cell that was written to (recall from the read test that we wrote to only one cell at `i=1` and `C=1`). Using the textbox on the left (in the red box shown below), you can enter the value you think is written to the highlighted cell.

![alt text](/docs/test_mode/images/value_textbox.png "Title")

<img src="../images/value_textbox" width="75%"/>

Using the recurrence relation try to enter the value written to the <span
style="background-color:#5c53a5">purple</span> cell on your own. Recall from the read test that we read cell `OPT[0, 1]` which has value `0` and ignore cell `OPT[0, 1 - c(1)]` because we are in a basecase. From our recurrence relation, that means we write the following value to the purple cell:
```
OPT[1, 1] = max(OPT[0, 1]) = 0
``` 
If you have not already done so, enter `0` into the textbox. The value test will be complete and the write test for the next timestep will be prompted.

## Navigating Test Mode
To exit test mode, click on the "Exit Testing Mode" button. The slider at the top of the screen will return to the screen so you can view other timesteps. Test mode can be started from any timestep. Try testing yourself on timestep `3` by moving the slider to the prior timestep (i.e. `2`) and then clicking the "Test Myself" button.
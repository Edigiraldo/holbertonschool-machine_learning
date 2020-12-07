0x01. Plotting
==============

#### 0\. Line Graph

Complete the following source code to plot `y` as a line graph:

-   `y` should be plotted as a solid red line
-   The x-axis should range from 0 to 10

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/9/664b2543b48ef4918687.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20201207%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20201207T213957Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=7efc80c6dfb1c69cd899693a62ce752d72d9da4d5961800416eacbf0c08020d7)

#### 1\. Scatter

Complete the following source code to plot `x ↦ y` as a scatter plot:

-   The x-axis should be labeled `Height (in)`
-   The y-axis should be labeled `Weight (lbs)`
-   The title should be `Men's Height vs Weight`
-   The data should be plotted as magenta points

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/9/1b143961d254e65afa2c.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20201207%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20201207T213957Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=82c8aae14942eea46faf176baf126300b9873d032ae4a1fc9fd81c9e6175123c)

#### 2\. Change of scale

Complete the following source code to plot `x ↦ y` as a line graph:

-   The x-axis should be labeled `Time (years)`
-   The y-axis should be labeled `Fraction Remaining`
-   The title should be `Exponential Decay of C-14`
-   The y-axis should be logarithmically scaled
-   The x-axis should range from 0 to 28650

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/9/2b6334feb069ae1b6014.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20201207%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20201207T213957Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=87b494048a9fc25b3e223d18f38c0790ca260995e0c2b5293283aafd6db174db)

#### 3\. Two is better than one

Complete the following source code to plot `x ↦ y1` and `x ↦ y2` as line graphs:

-   The x-axis should be labeled `Time (years)`
-   The y-axis should be labeled `Fraction Remaining`
-   The title should be `Exponential Decay of Radioactive Elements`
-   The x-axis should range from 0 to 20,000
-   The y-axis should range from 0 to 1
-   `x ↦ y1` should be plotted with a dashed red line
-   `x ↦ y2` should be plotted with a solid green line
-   A legend labeling `x ↦ y1` as `C-14` and `x ↦ y2` as `Ra-226` should be placed in the upper right hand corner of the plot

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/9/39eac4e8c8eb71469784.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20201207%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20201207T213957Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=8055d6d664b01f6d50447cfba410ddc6004a166ded50ad8dc00a499b38a51613)

#### 4\. Frequency

Complete the following source code to plot a histogram of student scores for a project:

-   The x-axis should be labeled `Grades`
-   The y-axis should be labeled `Number of Students`
-   The x-axis should have bins every 10 units
-   The title should be `Project A`
-   The bars should be outlined in black

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/9/10a48ad296d16ee8fb63.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20201207%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20201207T213957Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=e7ad933c3033bfd5744eeb7ee7984c47f621ab54d6ab69a5ae14cd0b7769969f)

#### 5\. All in One

Complete the following source code to plot all 5 previous graphs in one figure:

-   All axis labels and plot titles should have a font size of `x-small` (to fit nicely in one figure)
-   The plots should make a 3 x 2 grid
-   The last plot should take up two column widths (see below)
-   The title of the figure should be `All in One`

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/9/e58d423ffd060a779753.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20201207%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20201207T213957Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=0f4ff0b173aaa48647fa7c9ed8e3ccae05abcf6007f3427b1e23cd7f6022dc10)

#### 6\. Stacking Bars mandatory

Complete the following source code to plot a stacked bar graph:

-   `fruit` is a matrix representing the number of fruit various people possess
    -   The columns of `fruit` represent the number of fruit `Farrah`, `Fred`, and `Felicia` have, respectively
    -   The rows of `fruit` represent the number of `apples`, `bananas`, `oranges`, and `peaches`, respectively
-   The bars should represent the number of fruit each person possesses:
    -   The bars should be grouped by person, i.e, the horizontal axis should have one labeled tick per person
    -   Each fruit should be represented by a specific color:
        -   `apples` = red
        -   `bananas` = yellow
        -   `oranges` = orange (`#ff8000`)
        -   `peaches` = peach (`#ffe5b4`)
        -   A legend should be used to indicate which fruit is represented by each color
    -   The bars should be stacked in the same order as the rows of `fruit`, from bottom to top
    -   The bars should have a width of `0.5`
-   The y-axis should be labeled `Quantity of Fruit`
-   The y-axis should range from 0 to 80 with ticks every 10 units
-   The title should be `Number of Fruit per Person`

![](https://holbertonintranet.s3.amazonaws.com/uploads/medias/2018/9/e58d423ffd060a779753.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20201207%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20201207T213957Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=0f4ff0b173aaa48647fa7c9ed8e3ccae05abcf6007f3427b1e23cd7f6022dc10)

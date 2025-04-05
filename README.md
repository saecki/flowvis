# flowvis
Visualize a 2D vector field from the `flow.raw` file.

![image](https://github.com/user-attachments/assets/508b1986-7059-4c7a-a848-20d17fc31bbf)

## Build and run
- Install rust (using [rustup](https://rustup.rs/))
- Run `cargo run --release`

## Keymappings

| Key                       | Action                                                                   |
|---------------------------|--------------------------------------------------------------------------|
| <kbd>ctrl +</kbd>         | Zoom in                                                                  |
| <kbd>ctrl -</kbd>         | Zoom out                                                                 |
| <kbd>ctrl backspace</kbd> | Reset pan and zoom                                                       |
|                           |                                                                          |
| <kbd>space</kbd>          | Play/pause video playback                                                |
| <kbd>.</kbd>              | Step to next frame                                                       |
| <kbd>,</kbd>              | Step to previous frame                                                   |
| <kbd>[</kbd>              | Slow down playback                                                       |
| <kbd>]</kbd>              | Speed up playback                                                        |
|                           |                                                                          |
| <kbd>1</kbd>              | Show/hide the background velocity texture                                |
| <kbd>ctrl 1</kbd>         | Cycle through the color map for the background texture                   |
| <kbd>F</kbd>              | Enable/disable linear filtering of the background texture                |
|                           |                                                                          |
| <kbd>2</kbd>              | Show/hide stream lines                                                   |
| <kbd>ctrl 2</kbd>         | Cycle through the color map for stream lines                             |
| <kbd>I</kbd>              | Enable/disable interactive an stream line at the cursor position         |
| <kbd>M</kbd>              | Cycle the stream line method between `Euler`, `RK2` and `RK4`            |
| <kbd>shift M</kbd>        | Cycle the stream line method between `Euler`, `RK2` and `RK4` in reverse |
| <kbd>L</kbd>              | Spawn stream lines equally distributed on the left                       |
| <kbd>shift L</kbd>        | Spawn stream lines more concentrated towards the center left             |
| <kbd>delete</kbd>         | Remove all stream lines                                                  |
|                           |                                                                          |
| <kbd>3</kbd>              | Show/hide the arrow plot                                                 |
| <kbd>ctrl 3</kbd>         | Cycle through the color map for the arrow plot                           |
| <kbd>A</kbd>              | Increase arrow density                                                   |
| <kbd>shift A</kbd>        | Decrease arrow density                                                   |

## Mouse/touchpad control

| Key                              | Action                   |
|----------------------------------|--------------------------|
| <kbd>left-click</kbd>            | Add a stream line origin |
| <kbd>right-click</kbd>           | Remove a stream line     |
| <kbd>ctrl scroll</kbd>           | Zoom                     |
| <kbd>scroll</kbd>                | Pan                      |
| <kbd>shift scroll</kbd>          | Pan in x-direction       |
| <kbd>middle-click</kbd> and drag | Pan                      |


use crate::layout::{Rect, control_rect};
use crate::model::{ControlKind, EditorModel};
use font8x8::{BASIC_FONTS, UnicodeFonts};

const BACKGROUND: u32 = 0xff0b1020;
const PANEL: u32 = 0xff172033;
const PANEL_HOVER: u32 = 0xff202d45;
const TEXT: u32 = 0xfff5f7ff;
const MUTED: u32 = 0xffaab6cf;
const ACCENT: u32 = 0xff55d6ff;
const ACCENT_DARK: u32 = 0xff17667d;
const FOCUS: u32 = 0xffffd166;
const TRACK: u32 = 0xff303d57;

pub fn render(
    pixels: &mut [u32],
    width: u32,
    height: u32,
    model: &EditorModel,
    hovered: Option<usize>,
) {
    pixels.fill(BACKGROUND);
    if width == 0 || height == 0 {
        return;
    }
    draw_text(pixels, width, height, 22, 17, model.title(), TEXT, 2);
    draw_text(
        pixels,
        width,
        height,
        22,
        43,
        "Tab: focus   Arrows: adjust   Home/End: limits",
        MUTED,
        1,
    );

    for (index, spec) in model.specs().iter().enumerate() {
        let rect = control_rect(index, model.specs().len(), width, height);
        fill_rect(
            pixels,
            width,
            height,
            rect,
            if hovered == Some(index) {
                PANEL_HOVER
            } else {
                PANEL
            },
        );
        stroke_rect(
            pixels,
            width,
            height,
            rect,
            if model.focus() == index { FOCUS } else { TRACK },
            if model.focus() == index { 3 } else { 1 },
        );
        let value = model.value(index).unwrap_or(spec.default);
        let text_x = (rect.x + 14.0).round() as i32;
        let label_y = (rect.y + 12.0).round() as i32;
        draw_text(pixels, width, height, text_x, label_y, spec.name, MUTED, 1);
        draw_text(
            pixels,
            width,
            height,
            text_x,
            label_y + 19,
            &spec.display(value),
            TEXT,
            2,
        );

        let track = Rect {
            x: rect.x + 14.0,
            y: rect.y + rect.height - 20.0,
            width: (rect.width - 28.0).max(1.0),
            height: 7.0,
        };
        fill_rect(pixels, width, height, track, TRACK);
        let normalized = spec.normalized(value);
        let fill = Rect {
            width: track.width * normalized,
            ..track
        };
        fill_rect(pixels, width, height, fill, ACCENT);
        if matches!(spec.kind, ControlKind::Toggle) {
            let marker_width = 28.0;
            let marker = Rect {
                x: if normalized >= 0.5 {
                    track.x + track.width - marker_width
                } else {
                    track.x
                },
                y: track.y - 4.0,
                width: marker_width.min(track.width),
                height: track.height + 8.0,
            };
            fill_rect(
                pixels,
                width,
                height,
                marker,
                if normalized >= 0.5 {
                    ACCENT
                } else {
                    ACCENT_DARK
                },
            );
        }
    }
}

fn fill_rect(pixels: &mut [u32], width: u32, height: u32, rect: Rect, color: u32) {
    let left = rect.x.floor().max(0.0).min(f64::from(width)) as u32;
    let top = rect.y.floor().max(0.0).min(f64::from(height)) as u32;
    let right = (rect.x + rect.width).ceil().max(0.0).min(f64::from(width)) as u32;
    let bottom = (rect.y + rect.height)
        .ceil()
        .max(0.0)
        .min(f64::from(height)) as u32;
    for y in top..bottom {
        let row = y as usize * width as usize;
        for x in left..right {
            if let Some(pixel) = pixels.get_mut(row + x as usize) {
                *pixel = color;
            }
        }
    }
}

fn stroke_rect(
    pixels: &mut [u32],
    width: u32,
    height: u32,
    rect: Rect,
    color: u32,
    thickness: u32,
) {
    let thickness = f64::from(thickness);
    fill_rect(
        pixels,
        width,
        height,
        Rect {
            height: thickness,
            ..rect
        },
        color,
    );
    fill_rect(
        pixels,
        width,
        height,
        Rect {
            y: rect.y + rect.height - thickness,
            height: thickness,
            ..rect
        },
        color,
    );
    fill_rect(
        pixels,
        width,
        height,
        Rect {
            width: thickness,
            ..rect
        },
        color,
    );
    fill_rect(
        pixels,
        width,
        height,
        Rect {
            x: rect.x + rect.width - thickness,
            width: thickness,
            ..rect
        },
        color,
    );
}

#[allow(clippy::too_many_arguments)]
fn draw_text(
    pixels: &mut [u32],
    width: u32,
    height: u32,
    mut x: i32,
    y: i32,
    text: &str,
    color: u32,
    scale: u32,
) {
    let scale = scale.max(1) as i32;
    for character in text.chars() {
        let Some(glyph) = BASIC_FONTS.get(character) else {
            x += 8 * scale;
            continue;
        };
        for (row, bits) in glyph.iter().enumerate() {
            for column in 0..8 {
                if bits & (1 << column) == 0 {
                    continue;
                }
                for dy in 0..scale {
                    for dx in 0..scale {
                        let pixel_x = x + column * scale + dx;
                        let pixel_y = y + row as i32 * scale + dy;
                        if pixel_x >= 0
                            && pixel_y >= 0
                            && pixel_x < width as i32
                            && pixel_y < height as i32
                        {
                            let index = pixel_y as usize * width as usize + pixel_x as usize;
                            if let Some(pixel) = pixels.get_mut(index) {
                                *pixel = color;
                            }
                        }
                    }
                }
            }
        }
        x += 9 * scale;
        if x >= width as i32 {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{DisplayUnit, ModelError, ParameterSpec};

    const SPECS: &[ParameterSpec] = &[ParameterSpec {
        id: 1,
        name: "Mix",
        minimum: 0.0,
        maximum: 1.0,
        default: 1.0,
        step: 0.01,
        page_step: 0.1,
        kind: ControlKind::Continuous,
        unit: DisplayUnit::Percent,
    }];

    #[test]
    fn deterministic_software_frame_contains_focus_and_content() -> Result<(), ModelError> {
        let model = EditorModel::new("denoize", SPECS, &[0.5])?;
        let mut first = vec![0; 320 * 240];
        let mut second = vec![0; 320 * 240];
        render(&mut first, 320, 240, &model, None);
        render(&mut second, 320, 240, &model, None);
        assert_eq!(first, second);
        assert!(first.contains(&FOCUS));
        assert!(first.contains(&ACCENT));
        Ok(())
    }
}

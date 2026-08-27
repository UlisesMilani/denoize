#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Rect {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

impl Rect {
    pub fn contains(self, x: f64, y: f64) -> bool {
        x >= self.x && y >= self.y && x < self.x + self.width && y < self.y + self.height
    }
}

pub fn control_rect(index: usize, count: usize, width: u32, height: u32) -> Rect {
    let width = f64::from(width.max(1));
    let height = f64::from(height.max(1));
    let columns = if count > 4 { 2 } else { 1 };
    let rows = count.div_ceil(columns).max(1);
    let margin = (width * 0.035).clamp(12.0, 28.0);
    let header = (height * 0.19).clamp(58.0, 86.0);
    let gap = (width * 0.022).clamp(10.0, 20.0);
    let available_width = (width - margin * 2.0 - gap * (columns - 1) as f64).max(1.0);
    let available_height = (height - header - margin - gap * (rows - 1) as f64).max(1.0);
    let cell_width = available_width / columns as f64;
    let cell_height = available_height / rows as f64;
    let column = index % columns;
    let row = index / columns;
    Rect {
        x: margin + column as f64 * (cell_width + gap),
        y: header + row as f64 * (cell_height + gap),
        width: cell_width,
        height: cell_height,
    }
}

pub fn hit_test(count: usize, width: u32, height: u32, x: f64, y: f64) -> Option<usize> {
    (0..count).find(|index| control_rect(*index, count, width, height).contains(x, y))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_control_has_a_non_overlapping_hit_target() {
        let rects = (0..7)
            .map(|index| control_rect(index, 7, 640, 400))
            .collect::<Vec<_>>();
        for (index, rect) in rects.iter().enumerate() {
            assert!(rect.width >= 100.0);
            assert!(rect.height >= 40.0);
            assert_eq!(
                hit_test(
                    7,
                    640,
                    400,
                    rect.x + rect.width / 2.0,
                    rect.y + rect.height / 2.0
                ),
                Some(index)
            );
            for other in &rects[index + 1..] {
                let overlaps = rect.x < other.x + other.width
                    && rect.x + rect.width > other.x
                    && rect.y < other.y + other.height
                    && rect.y + rect.height > other.y;
                assert!(!overlaps);
            }
        }
    }
}

package com.betool.ui;


import javax.swing.border.LineBorder;
import java.awt.*;

public class CustomRoundedBorder extends LineBorder {
    private int radius; // 圆角半径

    public CustomRoundedBorder(Color color, int thickness, int radius) {
        super(color, thickness, true);
        this.radius = radius;
    }

    @Override
    public Insets getBorderInsets(Component c, Insets insets) {
        // 根据半径调整内边距，使圆角效果更明显
        insets.top = insets.bottom = radius / 2;
        insets.left = insets.right = radius / 2;
        return insets;
    }

    @Override
    public void paintBorder(Component c, Graphics g, int x, int y, int width, int height) {
        Graphics2D g2d = (Graphics2D) g;
        g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g2d.setColor(lineColor);
        g2d.setStroke(new BasicStroke(thickness));
        g2d.drawRoundRect(x, y, width - 1, height - 1, radius, radius);
    }
}
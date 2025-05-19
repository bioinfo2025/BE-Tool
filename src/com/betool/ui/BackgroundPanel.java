package com.betool.ui;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.IOException;

public class BackgroundPanel extends JPanel {

    public BufferedImage backgroundImage;
    public final boolean scaleToFit;

    public BackgroundPanel(String imageName, boolean scaleToFit) {
        this.scaleToFit = scaleToFit;
        try {
            // 加载背景图
            backgroundImage = ImageIO.read(BEToolUI.class.getResource(imageName));
            if (backgroundImage == null) {
                throw new IOException("无法加载背景图: " + imageName);
            }
        } catch (IOException e) {
            System.err.println("加载背景图时出错: " + e.getMessage());
            // 设置默认背景色
            setBackground(Color.LIGHT_GRAY);
        }
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        if (backgroundImage != null) {
            Graphics2D g2d = (Graphics2D) g.create();
            g2d.setRenderingHint(RenderingHints.KEY_INTERPOLATION,
                    RenderingHints.VALUE_INTERPOLATION_BILINEAR);

            if (scaleToFit) {
                // 缩放图片以适应面板
                g2d.drawImage(backgroundImage, 0, 0, getWidth(), getHeight(), this);
            } else {
                // 原始大小绘制
                g2d.drawImage(backgroundImage, 0, 0, this);
            }

            g2d.dispose();
        }
    }
}

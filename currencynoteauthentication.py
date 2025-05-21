import cv2
import numpy as np
import os
import re

def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Error loading image: {image_path}")
    image = cv2.resize(image, (600, 300))
    image = cv2.GaussianBlur(image, (5, 5), 0)
    return image

def get_orb_features(image):
    orb = cv2.ORB_create(1500)
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors

def match_features(des1, des2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    return good

def find_best_match(test_img_path, genuine_db_path):
    test_img = preprocess_image(test_img_path)
    kp2, des2 = get_orb_features(test_img)

    best_inliers = 0
    best_result_img = None
    best_note_status = "Counterfeit Note"

    match = re.search(r"\d+", os.path.basename(test_img_path))
    denomination = match.group() if match else None
    if not denomination:
        print("Denomination could not be extracted from file name.")
        return None, "Invalid File Name"

    denom_folder = os.path.join(genuine_db_path, denomination)
    if not os.path.exists(denom_folder):
        print(f"No folder found for denomination: {denomination}")
        return None, "Denomination Not Found"

    for file in os.listdir(denom_folder):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            ref_path = os.path.join(denom_folder, file)
            ref_img = preprocess_image(ref_path)
            kp1, des1 = get_orb_features(ref_img)

            if des1 is None or des2 is None or len(kp1) < 100 or len(kp2) < 100:
                continue

            good = match_features(des1, des2)

            if len(good) > 10:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                if mask is not None:
                    inliers = np.sum(mask)
                    match_percent = (inliers / len(good)) * 100 if good else 0

                    if inliers > 50 and match_percent > 60 and len(good) > 60:
                        status = "Genuine Note"
                    else:
                        status = "Counterfeit Note"

                    if inliers > best_inliers:
                        best_inliers = inliers
                        best_note_status = status

                        ref_color = cv2.cvtColor(ref_img, cv2.COLOR_GRAY2BGR)
                        test_color = cv2.cvtColor(test_img, cv2.COLOR_GRAY2BGR)

                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 1.2
                        thickness = 2
                        color = (0, 255, 0) if status == "Genuine Note" else (0, 0, 255)
                        label = status

                        if status == "Genuine Note":
                            draw_params = dict(matchColor=(0, 255, 0),
                                               singlePointColor=None,
                                               matchesMask=mask.ravel().tolist(),
                                               flags=2)

                            matched_img = cv2.drawMatches(ref_color, kp1, test_color, kp2, good, None, **draw_params)
                            pad_top = 60
                            matched_img = cv2.copyMakeBorder(matched_img, pad_top, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
                            cv2.putText(matched_img, label, (30, 45), font, font_scale, color, thickness, cv2.LINE_AA)
                            best_result_img = matched_img

                        else:
                            pad_top = 60
                            test_color = cv2.copyMakeBorder(test_color, pad_top, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
                            ref_color = cv2.copyMakeBorder(ref_color, pad_top, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0,0,0))
                            cv2.putText(test_color, label, (30, 45), font, font_scale, color, thickness, cv2.LINE_AA)

                            # Draw mismatched keypoints as red circles ( up to 30)
                            for i in range(min(30, len(kp2))):
                                pt = (int(kp2[i].pt[0]), int(kp2[i].pt[1]) + pad_top)
                                cv2.circle(test_color, pt, 6, (0, 0, 255), 2)

                            best_result_img = np.hstack((ref_color, test_color))

    return best_result_img, best_note_status

# === MAIN ===
test_image_path = "note_200.jpg"
genuine_notes_folder = "genuine notes"

try:
    result_img, status = find_best_match(test_image_path, genuine_notes_folder)

    print(f"\nNote Status: {status}")

    if result_img is not None:
        cv2.imshow("Currency Note Authentication", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No valid match found.")

except Exception as e:
    print(str(e))
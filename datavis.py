import argparse
import os
import shutil

import pandas as pd


def load_signs_df(train_csv_path):
	train_df = pd.read_csv(train_csv_path)
	if "Class" not in train_df.columns:
		raise ValueError("train CSV missing required column: Class")
	if "Corresponding Indexes" not in train_df.columns:
		raise ValueError("train CSV missing required column: Corresponding Indexes")
	if "cameraID" not in train_df.columns:
		raise ValueError("train CSV missing required column: cameraID")
	if "imageName" not in train_df.columns:
		raise ValueError("train CSV missing required column: imageName")
	signs_df = train_df[train_df["Class"] == "TrafficSign"].copy()
	# Keep a normalized identity column name for downstream logic.
	signs_df["objectID"] = signs_df["Corresponding Indexes"]
	return signs_df


def print_signs_stats(signs_df):
	print(f"Total sign images:     {len(signs_df)}")
	print(f"Total sign identities: {signs_df['objectID'].nunique()}")
	print("\nImages per camera:")
	print(signs_df["cameraID"].value_counts())
	print("\nIdentities per camera:")
	print(signs_df.groupby("cameraID")["objectID"].nunique())


def show_identity_samples(signs_df, output_csv, n_samples=5, random_state=42):
	annotation_template = []

	for oid, group in signs_df.groupby("objectID"):
		samples = group.sample(min(n_samples, len(group)), random_state=random_state)
		representative = samples.iloc[0]["imageName"]
		camera = group["cameraID"].value_counts().index[0]
		n_images = len(group)

		annotation_template.append(
			{
				"objectID": oid,
				"representative_img": representative,
				"primary_camera": camera,
				"n_images": n_images,
				"subclass": "",
			}
		)

	template_df = pd.DataFrame(annotation_template)
	template_df.to_csv(output_csv, index=False)

	print(f"Created {output_csv}")
	print("Open it, look at representative_img for each row,")
	print("and fill in the subclass column with your labels.")
	print(f"\n{len(template_df)} identities to annotate")

	return template_df


def sample_diverse_prototype_images(
	signs_df,
	subclass_object_ids,
	n_per_identity=2,
	n_per_camera=15,
	total=50,
	random_state=42,
):
	subclass_df = signs_df[signs_df["objectID"].isin(subclass_object_ids)].copy()

	print(
		f"Available: {len(subclass_df)} images, "
		f"{subclass_df['objectID'].nunique()} identities, "
		f"{subclass_df['cameraID'].nunique()} cameras"
	)

	selected = []
	for _, group in subclass_df.groupby("objectID"):
		sampled = group.sample(min(n_per_identity, len(group)), random_state=random_state)
		selected.append(sampled)

	if not selected:
		return pd.DataFrame(columns=subclass_df.columns)

	selected_df = pd.concat(selected, ignore_index=True)

	final = []
	camera_counts = {}

	selected_df = (
		selected_df.sample(frac=1, random_state=random_state)
		.reset_index(drop=True)
	)

	for _, row in selected_df.iterrows():
		cam = row["cameraID"]
		camera_counts[cam] = camera_counts.get(cam, 0)

		if camera_counts[cam] < n_per_camera:
			final.append(row)
			camera_counts[cam] += 1

		if len(final) >= total:
			break

	final_df = pd.DataFrame(final)

	print(f"\nSelected {len(final_df)} images:")
	if len(final_df) == 0:
		return final_df

	print(f"  Identities: {final_df['objectID'].nunique()}")
	print("  Camera distribution:")
	print(final_df["cameraID"].value_counts().to_string())
	id_sizes = final_df.groupby("objectID").size()
	print(
		"  Images per identity: "
		f"min={id_sizes.min()}, "
		f"max={id_sizes.max()}, "
		f"mean={id_sizes.mean():.1f}"
	)

	return final_df


def verify_diversity(prototype_selections):
	print("=" * 55)
	print("PROTOTYPE DIVERSITY REPORT")
	print("=" * 55)

	for subclass, df in prototype_selections.items():
		n_images = len(df)
		n_identities = df["objectID"].nunique() if n_images else 0
		n_cameras = df["cameraID"].nunique() if n_images else 0
		cam_dist = df["cameraID"].value_counts() if n_images else pd.Series(dtype=int)
		id_sizes = df.groupby("objectID").size() if n_images else pd.Series(dtype=int)

		print(f"\nSubclass: {subclass}")
		print(f"  Images:              {n_images}")
		print(f"  Unique identities:   {n_identities}")
		print(f"  Cameras covered:     {n_cameras}/3")
		print(f"  Camera breakdown:    {dict(cam_dist)}")
		print(f"  Max imgs/identity:   {id_sizes.max() if n_images else 0}")
		print(f"  Mean imgs/identity:  {id_sizes.mean() if n_images else 0:.1f}")

		if n_cameras < 2:
			print(
				f"  WARNING: Only {n_cameras} camera - "
				"prototype may be viewpoint-biased"
			)
		if n_images and id_sizes.max() > 5:
			print(
				f"  WARNING: One identity has {id_sizes.max()} images - "
				"consider capping at 2-3"
			)
		if n_identities < 10:
			print(
				f"  WARNING: Only {n_identities} identities - "
				"prototype may overfit to specific instances"
			)

		if n_cameras >= 2 and (not n_images or id_sizes.max() <= 3) and n_identities >= 10:
			print("  Diversity looks good")

	print("\n" + "=" * 55)


def copy_images(selected_df, image_dir, output_dir):
	os.makedirs(output_dir, exist_ok=True)

	for _, row in selected_df.iterrows():
		src = os.path.join(image_dir, row["imageName"])
		dst = os.path.join(output_dir, os.path.basename(row["imageName"]))

		if not os.path.exists(src):
			print(f"Missing image: {src}")
			continue

		shutil.copy2(src, dst)


def sample_from_annotations(
	signs_df,
	annotations_csv,
	prototypes_dir,
	image_dir,
	n_per_identity=2,
	n_per_camera=15,
	total=50,
	copy_images_flag=False,
):
	annotations = pd.read_csv(annotations_csv)
	annotations = annotations[annotations["subclass"].fillna("") != ""]

	print("Subclass distribution across identities:")
	print(annotations["subclass"].value_counts())

	prototype_selections = {}

	for subclass, group in annotations.groupby("subclass"):
		object_ids = group["objectID"].tolist()

		selected = sample_diverse_prototype_images(
			signs_df,
			subclass_object_ids=object_ids,
			n_per_identity=n_per_identity,
			n_per_camera=n_per_camera,
			total=total,
		)

		prototype_selections[subclass] = selected

		os.makedirs(prototypes_dir, exist_ok=True)
		output_csv = os.path.join(prototypes_dir, f"prototype_images_{subclass}.csv")
		selected.to_csv(output_csv, index=False)
		print(f"\nSubclass {subclass}: saved {len(selected)} images to {output_csv}")

		if copy_images_flag:
			subclass_dir = os.path.join(prototypes_dir, f"subclass_{subclass}")
			copy_images(selected, image_dir, subclass_dir)

	return prototype_selections


def parse_args():
	parser = argparse.ArgumentParser(
		description="Sample diverse traffic sign prototypes across identities and cameras."
	)
	parser.add_argument(
		"--dataset-root",
		default=".",
		help="Dataset root containing train_classes.csv and image_train.",
	)
	parser.add_argument(
		"--train-csv",
		default="train_classes.csv",
		help="Train CSV filename inside dataset root.",
	)
	parser.add_argument(
		"--image-dir",
		default="image_train",
		help="Image directory inside dataset root.",
	)
	parser.add_argument(
		"--prototypes-dir",
		default="prototypes",
		help="Output folder for prototype CSVs and optional images.",
	)
	parser.add_argument(
		"--make-template",
		action="store_true",
		help="Create sign_annotation_template.csv for manual subclass labels.",
	)
	parser.add_argument(
		"--annotations-csv",
		default="sign_annotation_template.csv",
		help="CSV with subclass labels per identity.",
	)
	parser.add_argument(
		"--sample",
		action="store_true",
		help="Sample prototypes from annotations CSV.",
	)
	parser.add_argument(
		"--verify",
		action="store_true",
		help="Print diversity report after sampling.",
	)
	parser.add_argument(
		"--n-per-identity",
		type=int,
		default=2,
		help="Max images per identity.",
	)
	parser.add_argument(
		"--n-per-camera",
		type=int,
		default=15,
		help="Max images per camera.",
	)
	parser.add_argument(
		"--total",
		type=int,
		default=50,
		help="Total images per subclass prototype.",
	)
	parser.add_argument(
		"--copy-images",
		action="store_true",
		help="Copy selected images into prototypes subfolders.",
	)
	return parser.parse_args()


def main():
	args = parse_args()

	train_csv_path = os.path.join(args.dataset_root, args.train_csv)
	image_dir = os.path.join(args.dataset_root, args.image_dir)

	if not os.path.exists(train_csv_path):
		raise FileNotFoundError(f"Train CSV not found: {train_csv_path}")
	if not os.path.isdir(image_dir):
		raise FileNotFoundError(f"Image dir not found: {image_dir}")

	signs_df = load_signs_df(train_csv_path)
	print_signs_stats(signs_df)

	if args.make_template:
		show_identity_samples(signs_df, args.annotations_csv)

	prototype_selections = {}
	if args.sample:
		prototype_selections = sample_from_annotations(
			signs_df,
			args.annotations_csv,
			args.prototypes_dir,
			image_dir,
			n_per_identity=args.n_per_identity,
			n_per_camera=args.n_per_camera,
			total=args.total,
			copy_images_flag=args.copy_images,
		)

	if args.verify and prototype_selections:
		verify_diversity(prototype_selections)


if __name__ == "__main__":
	main()
